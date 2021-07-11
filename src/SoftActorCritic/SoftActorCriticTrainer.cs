namespace LostTech.Torch.RL.SoftActorCritic {
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using TorchSharp;
    using TorchSharp.NN;
    using TorchSharp.Tensor;

    public class SoftActorCriticTrainer {
        public ActorCritic ActorCritic { get; }
        public ActorCritic TargetActorCritic { get; }
        public Optimizer QOptimizer { get; }
        public Optimizer PiOptimizer { get; }
        public Device? Device { get; }
        /// <summary>
        /// How much to discount previous rewards (factor per step).
        /// </summary>
        public float RewardDiscount { get; set; } = 0.99f;
        /// <summary>
        /// <c>(0..1]</c> SAC uses Polyak averaging to smooth training. This parameter
        /// affects how fast stable "target" policy is moving towards recent learnings.
        /// 
        /// If the value is too low, the algorithm will train slowly.
        /// 
        /// If the value is too high, the learning process might be unstable.
        /// </summary>
        public float TargetUpdateFactor { get; set; } = 0.995f;
        /// <summary>Affects how random agent's actions will be.</summary>
        public float BehaviorRandomness { get; set; } = 0.2f;

        TorchTensor[] QParams { get; }

        const string OptimizerFactoryMustNotReturnNull = "Optimizer factory must return non-null value";

        public SoftActorCriticTrainer(Func<ActorCritic> actorCriticFactory,
                                      Func<IEnumerable<TorchTensor>, Optimizer> qOptimizerFactory,
                                      Func<IEnumerable<TorchTensor>, Optimizer> piOptimizerFactory) {
            this.ActorCritic = actorCriticFactory();
            this.TargetActorCritic = actorCriticFactory();
            foreach (var (var, targetVar) in Enumerable.Zip(this.ActorCritic.parameters(), this.TargetActorCritic.parameters(), Tuple)) {
                // only updated using polyak averaging
                targetVar.requires_grad = false;
                targetVar.copy_(source: var);
            }

            this.Device = this.ActorCritic.Actor.Device;

            this.QParams = this.ActorCritic.Q1.parameters().Concat(this.ActorCritic.Q2.parameters()).ToArray();

            this.QOptimizer = qOptimizerFactory(this.QParams) ?? throw new ArgumentException(OptimizerFactoryMustNotReturnNull);
            this.PiOptimizer = piOptimizerFactory(this.ActorCritic.Actor.parameters()) ?? throw new ArgumentException(OptimizerFactoryMustNotReturnNull);
        }

        TorchTensor QLoss(ReplayBufferEntry historyBatch) {
            TorchTensor obs_act = new[] { historyBatch.Observation, historyBatch.Action }.cat(-1);
            TorchTensor q1 = this.ActorCritic.Q1.forward(obs_act).squeeze(-1);
            TorchTensor q2 = this.ActorCritic.Q2.forward(obs_act).squeeze(-1);

            // Bellman backup for Q functions
            TorchTensor backup;
            using (new AutoGradMode(false)) {
                // target actions come from *current* policy
                TorchTensor targetAction = this.ActorCritic.Actor.forward(historyBatch.NewObservation, out TorchTensor targetActionLogProb);

                // target Q-values
                TorchTensor newObs_targetAct = new[] { historyBatch.NewObservation, targetAction }.cat(-1);
                TorchTensor q1pi = this.TargetActorCritic.Q1.forward(newObs_targetAct).squeeze(-1);
                TorchTensor q2pi = this.TargetActorCritic.Q2.forward(newObs_targetAct).squeeze(-1);

                TorchTensor qPi = q1pi.minimum(q2pi);

                backup = historyBatch.Reward + this.RewardDiscount * (-historyBatch.Done + 1) * (qPi - this.BehaviorRandomness * targetActionLogProb);
            }

            TorchTensor lossQ1 = (q1 - backup).pow(2).mean();
            TorchTensor lossQ2 = (q2 - backup).pow(2).mean();

            return lossQ1 + lossQ2;
        }

        TorchTensor PiLoss(TorchTensor observation) {
            TorchTensor pi = this.ActorCritic.Actor.forward(observation, out TorchTensor logProb);
            TorchTensor obs_act = new[] { observation, pi }.cat(-1);
            TorchTensor q1pi = this.ActorCritic.Q1.forward(obs_act).squeeze(-1);
            TorchTensor q2pi = this.ActorCritic.Q2.forward(obs_act).squeeze(-1);
            TorchTensor qPi = q1pi.minimum(q2pi);

            // Entropy-regularized policy loss
            return (this.BehaviorRandomness * logProb - qPi).mean();
        }

        public TrainResult Train(ReplayBufferEntry historyBatch) {
            if (this.Device is not null)
                historyBatch = historyBatch.To(this.Device);

            this.QOptimizer.zero_grad();
            var lossQ = this.QLoss(historyBatch);
            lossQ.backward();
            this.QOptimizer.step();

            foreach (var param in this.QParams)
                param.requires_grad = false;

            this.PiOptimizer.zero_grad();
            var lossPi = this.PiLoss(historyBatch.Observation);
            lossPi.backward();
            this.PiOptimizer.step();

            foreach (var param in this.QParams)
                param.requires_grad = true;

            using var noGrad = new AutoGradMode(false);
            foreach (var (var, targetVar) in Enumerable.Zip(this.ActorCritic.parameters(), this.TargetActorCritic.parameters(), Tuple)) {
                targetVar.mul_(this.TargetUpdateFactor);
                targetVar.add_(var * (1 - this.TargetUpdateFactor));
            }

            return new TrainResult {
                LossQ = lossQ.cpu().mean().ToScalar().ToSingle(),
                LossPi = lossPi.cpu().mean().ToScalar().ToSingle(),
            };
        }

        static (T1, T2) Tuple<T1, T2>(T1 v1, T2 v2) => (v1, v2);

        public class TrainResult {
            public float LossQ { get; init; }
            public float LossPi { get; init; }

            public override string ToString()
                => $"LossQ: {this.LossQ}  LossPi: {this.LossPi}";
        }
    }
}
