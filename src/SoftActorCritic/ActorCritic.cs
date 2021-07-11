namespace LostTech.Torch.RL.SoftActorCritic {
    using System;
    using System.Linq;
    using TorchSharp;
    using TorchSharp.NN;
    using TorchSharp.Tensor;

    /// <summary>
    /// Encapsulates neural networks, that represent Soft Actor-Critic-based agent
    /// </summary>
    public class ActorCritic : CustomModule {
        /// <summary>
        /// Networks that decide what agent is going to do in the environment
        /// </summary>
        public Actor Actor { get; }
        /// <summary>
        /// Network, that estimates future agent rewards.
        /// </summary>
        /// <seealso cref="Q2"/>
        public Module Q1 { get; }
        /// <summary>
        /// Network, that estimate future agent rewards.
        /// </summary>
        /// <seealso cref="Q1"/>
        public Module Q2 { get; }

        public ActorCritic(Actor actor, Module q1, Module q2) : base("ActorCritic") {
            this.Actor = actor ?? throw new ArgumentNullException(nameof(actor));
            this.Q1 = q1 ?? throw new ArgumentNullException(nameof(q1));
            this.Q2 = q2 ?? throw new ArgumentNullException(nameof(q2));

            if (actor.Device is { } device) {
                this.Q1 = this.Q1.to(device);
                this.Q2 = this.Q2.to(device);
            }

            this.RegisterModule(nameof(this.Actor), this.Actor);
            this.RegisterModule(nameof(this.Q1), this.Q1);
            this.RegisterModule(nameof(this.Q2), this.Q2);
        }

        public override TorchTensor forward(TorchTensor t) => throw new NotSupportedException();

        public Span<float> Act(float[] observation, bool deterministic = false) {
            if (observation is null) throw new ArgumentNullException(nameof(observation));

            using var noGrad = new AutoGradMode(false);
            var observationTensor = Float32Tensor.from(observation,
                                                       dimensions: new[] { observation.Length, 1L });
            if (this.Actor.Device is { } device)
                observationTensor = observationTensor.to(device);
            var action = this.Actor.forward(observationTensor,
                                            deterministic: deterministic);
            return action.cpu().Data<float>();
        }
    }
}
