namespace LostTech.Torch.RL;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using global::RL.Envs;
using LostTech.Torch.RL.SoftActorCritic;

using TorchSharp;

using Xunit;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;

public class ActorCriticTrainingTest {
    [Fact]
    public void TrainsOnRepeatObservation() {
        var env = new RepeatObservation();
        torch.random.manual_seed(112);

        var replayBuffer = new ReplayBuffer(observationDimensions: 1, actionDimensions: 1,
                                            size: 64 * 1024, batchSize: 1);

        ActorCritic ActorCriticFactory() {
            // LeakyRELU does not work o-O
            static Module activation() => ReLU(inPlace: false);

            const int backInner = 16;


            var backbone = Sequential(("bb_h1", Linear(1, backInner)),
                                      ("bb_h1_act", activation()),
                                      ("bb_h2", Linear(backInner, backInner)),
                                      ("bb_h2_act", activation()),
                                      ("bb_h3", Linear(backInner, backInner)),
                                      ("bb_h3_act", activation()));

            var actor = new Actor(
                backbone: backbone,
                action: Linear(backInner, 1),
                actionDistribution: Linear(backInner, 1),
                actionMin: env.ActionSpace.Low, actionMax: env.ActionSpace.High);

            int qInputSize = 1 + 1;
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
        var trainer = new SoftActorCriticTrainer(ActorCriticFactory,
            qOptimizerFactory: @params => Adam(@params),
            piOptimizerFactory: @params => Adam(@params));

        const int totalSteps = 64 * 1024;
        const int randomSteps = 128;

        const int updateAfter = randomSteps;
        const int updateEvery = 1024;

        float aiAction = 0;

        float observation = env.Reset();
        float totalReward = 0;

        var random = new Random();

        for (int stepN = 0; stepN < totalSteps; stepN++) {
            using var _ = torch.NewDisposeScope();
            float action = stepN > randomSteps
                ? trainer.ActorCritic.Act(new []{observation}, deterministic: stepN % 28 == 0)[0]
                : (float)random.NextDouble();
            aiAction += MathF.Abs(action);

            var stepResult = env.Step(action);

            float newObservation = stepResult.Observation;
            float reward = stepResult.Reward;
            totalReward += reward;

            var recording = new ReplayBufferEntry(
                observation: unsqueeze(tensor(observation), 0),
                newObservation: unsqueeze(tensor(newObservation), 0),
                action: unsqueeze(tensor(action), 0),
                reward: unsqueeze(tensor(reward), 0),
                done: torch.zeros(1)
            );
            replayBuffer.Store(recording);
            recording.Dispose();

            observation = newObservation;

            if (stepN >= updateAfter && stepN % updateEvery == updateEvery - 1) {
                const int trainBatches = 128;
                var trainResults = new List<SoftActorCriticTrainer.TrainResult>();
                for (int batchN = 0; batchN < trainBatches; batchN++) {
                    const int batchSize = 128;
                    var batch = replayBuffer.SampleBatch(batchSize);
                    var result = trainer.Train(batch);
                    trainResults.Add(result);
                }

                Trace.WriteLine(new SoftActorCriticTrainer.TrainResult {
                    LossQ = trainResults.Select(r => r.LossQ).Average(),
                    LossPi = trainResults.Select(r => r.LossPi).Average(),
                });
                Trace.WriteLine($"avg. reward: {totalReward / updateEvery}");
                totalReward = 0;

                aiAction = 0;
                observation = env.Reset();
            }
        }

        float actionValidation = trainer.ActorCritic.Act(new []{observation}, deterministic: true)[0];
        float avgDiff = MathF.Abs(observation - actionValidation);
        Assert.True(avgDiff < 0.1);
    }
}
