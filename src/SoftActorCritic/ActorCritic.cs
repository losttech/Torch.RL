namespace LostTech.Torch.RL.SoftActorCritic;

using System;
using System.Linq;

using TorchSharp;
using TorchSharp.Utils;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

/// <summary>
/// Encapsulates neural networks, that represent Soft Actor-Critic-based agent
/// </summary>
public class ActorCritic : Module {
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

        this.RegisterComponents();
    }

    public override Tensor forward(Tensor t) => throw new NotSupportedException();

    public TensorAccessor<float> Act(IList<float> observation, bool deterministic = false) {
        if (observation is null) throw new ArgumentNullException(nameof(observation));

        using var noGrad = torch.no_grad();
        var observationTensor = torch.tensor(observation,
                                             dimensions: new[] { 1L, observation.Count });
        if (this.Actor.Device is { } device)
            observationTensor = observationTensor.to(device);
        var action = this.Actor.forward(observationTensor,
                                        deterministic: deterministic);
        return action.cpu().data<float>();
    }
}
