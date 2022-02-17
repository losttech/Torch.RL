using LostTech.Gradient;
using LostTech.Torch.RL.SoftActorCritic;

using numpy;

using PyGym;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;

// PyGym_ENV environment variable must be set to conda:your-env-name
// before launching this you need to create conda environment your-env-name
// and install numpy, gym, and pyglet there
GradientEngine.UseEnvironmentFromVariable("PyGym_ENV");

using var env = Gym.Make("CartPole-v1");
ndarray<float> observation = env.Reset().astype(np.float32_fn);
int observationSize = (int)env.ObservationSpace.shape[0];
int actionSize = (int)env.ActionSpace.shape.__len__() switch {
    0 => 1,
    1 => env.ActionSpace.shape[0],
    _ => throw new NotImplementedException(),
};

var replayBuffer = new ReplayBuffer(observationDimensions: observation.Length,
                                    actionDimensions: 1,
                                    size: 64 * 1024, batchSize: 1);

var trainer = new SoftActorCriticTrainer(ActorCriticFactory,
    qOptimizerFactory: @params => Adam(@params),
    piOptimizerFactory: @params => Adam(@params));

const int totalSteps = 32 * 1024;
const int randomSteps = 0;

const int updateAfter = randomSteps;
const int updateEvery = 1024;

float aiAction = 0;

float runningAvgReward = 0, episodeReward = 0;
int episodeSteps = 0;

var random = new Random();

for (int stepN = 0; stepN < totalSteps; stepN++) {
    using var _ = NewDisposeScope();
    // float action = random.Next(2);
    // if (stepN > randomSteps) {
    float action = trainer.ActorCritic.Act(observation.ToArray1D(), deterministic: stepN % 28 == 0)[0];
    //}
    aiAction += MathF.Abs(action);

    var stepResult = env.Step(random.NextDouble() * 2 > action ? 0 : 1);

    ndarray<float> newObservation = stepResult[0].astype(np.float32_fn);
    float reward = (float)stepResult[1];
    episodeSteps++;
    episodeReward += reward;
    bool done = stepResult[2];
    if (done) {
        episodeSteps = 0;
        if (runningAvgReward == 0) runningAvgReward = episodeReward;
        runningAvgReward = runningAvgReward * 0.95f + episodeReward * 0.05f;
        episodeReward = 0;
    }

    var recording = new ReplayBufferEntry(
        observation: unsqueeze(tensor(observation.ToArray1D()), 0),
        newObservation: unsqueeze(tensor(newObservation.ToArray1D()), 0),
        action: unsqueeze(tensor(action), 0),
        reward: unsqueeze(tensor(reward), 0),
        done: done ? ones(1) : zeros(1)
    );
    replayBuffer.Store(recording);
    recording.Dispose();

    observation = done ? env.Reset().astype(np.float32_fn) : newObservation;

    if (stepN >= updateAfter) {
        env.Render();
    }

    if (stepN >= updateAfter && stepN % updateEvery == updateEvery - 1) {
        const int trainBatches = 128;
        var trainResults = new List<SoftActorCriticTrainer.TrainResult>();
        for (int batchN = 0; batchN < trainBatches; batchN++) {
            const int batchSize = 128;
            var batch = replayBuffer.SampleBatch(batchSize);
            var result = trainer.Train(batch);
            trainResults.Add(result);
        }

        Console.WriteLine(new SoftActorCriticTrainer.TrainResult {
            LossQ = trainResults.Select(r => r.LossQ).Average(),
            LossPi = trainResults.Select(r => r.LossPi).Average(),
        });
        Console.WriteLine($"avg. reward: {runningAvgReward} A.I. action: {aiAction / updateEvery}");

        aiAction = 0;
        observation = env.Reset().astype(np.float32_fn);
    }
}

ActorCritic ActorCriticFactory() {
    // LeakyRELU does not work o-O
    static Module activation() => ReLU(inPlace: false);

    const int backInner = 16;


    var backbone = Sequential(Linear(observationSize, backInner),
                              activation(),
                              Linear(backInner, backInner),
                              activation(),
                              Linear(backInner, backInner),
                              activation());

    var actor = new Actor(
        backbone: backbone,
        action: Linear(backInner, 1),
        actionDistribution: Linear(backInner, 1),
        actionMin: 0, actionMax: (int)env.ActionSpace.n - 0.0001f);

    int qInputSize = observationSize + actionSize;
    const int qInner = 16;
    Module Q() => Sequential(
            //("q1_back", backbone),
            ("h1", Linear(qInputSize, qInner)),
            ("h1_act", activation()),
            ("h2", Linear(qInner, qInner)),
            ("h2_act", activation()),
            ("h3", Linear(qInner, qInner)),
            ("h3_act", activation()),
            ("out", Linear(qInner, 1))
        );

    var ac = new ActorCritic(actor, q1: Q(), q2: Q());

    return ac;
}