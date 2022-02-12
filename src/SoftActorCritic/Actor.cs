namespace LostTech.Torch.RL.SoftActorCritic;

using System;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class Actor : Module {
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

        this.RegisterComponents();

        this.ActionMin = actionMin;
        this.ActionMax = actionMax;

        if (device is not null)
            this.to(device);
        this.Device = device;
    }

    const float LogStdMax = 2;
    const float LogStdMin = -20;

    public sealed override Tensor forward(Tensor observation) => this.forward(observation, deterministic: false);
    public virtual Tensor forward(Tensor observation, out Tensor logProb) {
        this.Mu_LogStd(observation, out var mu, out var logStd);

        var piDistribution = new NormalDistribution(mu, logStd.exp());

        var piAction = piDistribution.Sample(device: this.Device);
        logProb = piDistribution.LogProb(piAction).sum(dimensions: new long[] { -1 });
        logProb -= (2 * (-piAction + MathF.Log(2) - (-2 * piAction).softplus()))
            .sum(dimensions: new long[] { 1 });

        return this.MakeAction(piAction);
    }
    public virtual Tensor forward(Tensor observation, bool deterministic) {
        this.Mu_LogStd(observation, out var mu, out var logStd);

        var piAction = deterministic ? mu : new NormalDistribution(mu, logStd.exp()).Sample(device: this.Device);

        return this.MakeAction(piAction);
    }

    void Mu_LogStd(Tensor observation, out Tensor mu, out Tensor logStd) {
        var backboneOut = this.Backbone.forward(observation);
        mu = this.Action.forward(backboneOut);
        logStd = this.ActionDistribution.forward(backboneOut);
        logStd = logStd.clamp(min: LogStdMin, max: LogStdMax);
    }

    Tensor MakeAction(Tensor piAction) {
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
