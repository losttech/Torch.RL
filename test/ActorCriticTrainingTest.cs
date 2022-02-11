namespace LostTech.Torch.RL {
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using LostTech.AI.RL;
    using LostTech.Torch.RL.SoftActorCritic;

    using TorchSharp;

    using Xunit;

    using static TorchSharp.torch;
    using static TorchSharp.torch.nn;
    using static TorchSharp.torch.optim;

    public class ActorCriticTrainingTest {
        [Fact]
        public void TrainsOnRepeatObservation() {
            var env = new RepeatObservationEnvironment(Environment.ProcessorCount);
            random.manual_seed(112);

            var replayBuffer = new ReplayBuffer(observationDimensions: 1, actionDimensions: 1,
                                                size: 64 * 1024, batchSize: env.AgentCount);

            ActorCritic ActorCriticFactory() {
                // LeakyRELU does not work o-O
                static Module activation() => ReLU(inPlace: false);

                const int backInner = 16;


                var backbone = Sequential(("bb_h1", Linear(env.ObservationSize, backInner)),
                                          ("bb_h1_act", activation()),
                                          ("bb_h2", Linear(backInner, backInner)),
                                          ("bb_h2_act", activation()),
                                          ("bb_h3", Linear(backInner, backInner)),
                                          ("bb_h3_act", activation()));

                var actor = new Actor(
                    backbone: backbone,
                    action: Linear(backInner, 1),
                    actionDistribution: Linear(backInner, 1),
                    actionMin: env.ActionMin, actionMax: env.ActionMax);

                int qInputSize = env.ObservationSize + env.ActionSize;
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

            const int totalSteps = 8*1024;
            const int randomSteps = 128;

            const int updateAfter = randomSteps;
            const int updateEvery = 1024;

            float aiAction = 0;

            var stepResult = env.GetStepResult(null);

            float[] observation = stepResult.Observation;
            float avgReward = 0;

            for (int stepN = 0; stepN < totalSteps; stepN++) {
                using var _ = torch.NewDisposeScope();
                float[] action = stepN > randomSteps
                    ? trainer.ActorCritic.Act(observation, deterministic: stepN % 28 == 0).ToArray()
                    : env.SampleAction();
                aiAction += action.Sum();

                env.SetActions(null, action);
                env.Step();

                stepResult = env.GetStepResult(null);

                float[] newObservation = stepResult.Observation;
                float[] reward = stepResult.Reward;
                avgReward += reward.Average();

                var recording = new ReplayBufferEntry(
                    observation: tensor(observation),
                    newObservation: torch.tensor(newObservation),
                    action: torch.tensor(action),
                    reward: torch.tensor(reward),
                    done: torch.zeros(env.AgentCount)
                );
                replayBuffer.Store(recording);
                recording.Dispose();

                newObservation.CopyTo(observation, 0);

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
                    Trace.WriteLine($"avg. reward: {avgReward / updateEvery}");
                    avgReward = 0;

                    aiAction = 0;
                    env.Reset();
                    stepResult = env.GetStepResult(null);
                    observation = stepResult.Observation;
                }
            }

            float[] actionValidation = trainer.ActorCritic.Act(stepResult.Observation, deterministic: true).ToArray();
            float avgDiff = stepResult.Observation.Zip(actionValidation, (o, a) => MathF.Abs(o - a)).Average();
            Assert.True(avgDiff < 0.1);
        }
    }
}
