namespace LostTech.Torch.RL.SoftActorCritic;

using System;

using static TorchSharp.torch;

public class ReplayBufferEntry : IDisposable {
    public ReplayBufferEntry(Tensor observation, Tensor newObservation, Tensor action, Tensor reward, Tensor done) {
        this.Observation = observation ?? throw new ArgumentNullException(nameof(observation));
        this.NewObservation = newObservation ?? throw new ArgumentNullException(nameof(newObservation));
        this.Action = action ?? throw new ArgumentNullException(nameof(action));
        this.Reward = reward ?? throw new ArgumentNullException(nameof(reward));
        this.Done = done ?? throw new ArgumentNullException(nameof(done));
    }

    public Tensor Observation { get; init; }
    public Tensor NewObservation { get; init; }
    public Tensor Action { get; init; }
    public Tensor Reward { get; init; }
    public Tensor Done { get; init; }

    public ReplayBufferEntry To(Device device)
        => new(
            observation: this.Observation.to(device),
            newObservation: this.NewObservation.to(device),
            action: this.Action.to(device),
            reward: this.Reward.to(device),
            done: this.Done.to(device));

    public void Dispose() {
        foreach (var tensor in new[] { this.Observation, this.Action, this.Reward, this.Done, this.NewObservation }) tensor.Dispose();
    }
}
