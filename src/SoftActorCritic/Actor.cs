namespace LostTech.Torch.RL {
    using System;
    using TorchSharp;
    using TorchSharp.NN;
    using TorchSharp.Tensor;

    public class Actor : CustomModule {
        public Module Backbone { get; }
        public Module Action { get; }
        public Module ActionDistribution { get; }
        public float ActionMin { get; }
        public float ActionMax { get; }
        public Device? Device { get; }

        public Actor(Module backbone, Module action, Module actionDistribution,
                     float actionMin = -1, float actionMax = +1,
                     Device? device = null,
                     string name = "actor") : base(name) {
            this.Backbone = backbone ?? throw new ArgumentNullException(nameof(backbone));
            this.Action = action ?? throw new ArgumentNullException(nameof(action));
            this.ActionDistribution = actionDistribution ?? throw new ArgumentNullException(nameof(actionDistribution));

            this.RegisterModule(nameof(this.Backbone), this.Backbone);
            this.RegisterModule(nameof(this.Action), this.Action);
            this.RegisterModule(nameof(this.ActionDistribution), this.ActionDistribution);

            this.ActionMin = actionMin;
            this.ActionMax = actionMax;

            if (device is not null)
                this.to(device);
            this.Device = device;
        }

        const float LogStdMax = 2;
        const float LogStdMin = -20;

        public sealed override TorchTensor forward(TorchTensor observation) => this.forward(observation, deterministic: false);
        public virtual TorchTensor forward(TorchTensor observation, out TorchTensor logProb) {
            this.Mu_LogStd(observation, out var mu, out var logStd);

            var piDistribution = new NormalDistribution(mu, logStd.exp());

            var piAction = piDistribution.Sample(device: this.Device);
            logProb = piDistribution.LogProb(piAction).sum(dimensions: new long[] { -1 });
            logProb -= (2 * (-piAction + MathF.Log(2) - (-2 * piAction).softplus()))
                .sum(dimensions: new long[] { 1 });

            return this.MakeAction(piAction);
        }
        public virtual TorchTensor forward(TorchTensor observation, bool deterministic) {
            this.Mu_LogStd(observation, out var mu, out var logStd);

            var piAction = deterministic ? mu : new NormalDistribution(mu, logStd.exp()).Sample(device: this.Device);

            return this.MakeAction(piAction);
        }

        void Mu_LogStd(TorchTensor observation, out TorchTensor mu, out TorchTensor logStd) {
            var backboneOut = this.Backbone.forward(observation);
            mu = this.Action.forward(backboneOut);
            logStd = this.ActionDistribution.forward(backboneOut);
            logStd = logStd.clamp(min: LogStdMin, max: LogStdMax);
        }

        TorchTensor MakeAction(TorchTensor piAction) {
            var action = piAction.tanh();

            if (float.IsNaN(this.ActionMin)) throw new ArgumentException(nameof(this.ActionMin));
            if (float.IsNaN(this.ActionMax)) throw new ArgumentException(nameof(this.ActionMax));
            if (float.IsInfinity(this.ActionMin) || float.IsInfinity(this.ActionMax))
                throw new NotImplementedException("Unlimited action");
            if (this.ActionMax <= this.ActionMin)
                throw new ArgumentException($"{nameof(this.ActionMax)} must be greater than {nameof(this.ActionMin)}");

            action = (action + 1) // 0..2
                     * ((this.ActionMax - this.ActionMin) / 2) // 0 .. max-min
                     + this.ActionMin; // min..max

            return action;
        }
    }
}
