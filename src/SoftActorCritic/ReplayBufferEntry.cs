namespace LostTech.Torch.RL {
    using System;
    using TorchSharp;
    using TorchSharp.Tensor;

    public class ReplayBufferEntry : IDisposable {
        public ReplayBufferEntry(TorchTensor observation, TorchTensor newObservation, TorchTensor action, TorchTensor reward, TorchTensor done) {
            this.Observation = observation ?? throw new ArgumentNullException(nameof(observation));
            this.NewObservation = newObservation ?? throw new ArgumentNullException(nameof(newObservation));
            this.Action = action ?? throw new ArgumentNullException(nameof(action));
            this.Reward = reward ?? throw new ArgumentNullException(nameof(reward));
            this.Done = done ?? throw new ArgumentNullException(nameof(done));
        }

        public TorchTensor Observation { get; init; }
        public TorchTensor NewObservation { get; init; }
        public TorchTensor Action { get; init; }
        public TorchTensor Reward { get; init; }
        public TorchTensor Done { get; init; }

        public ReplayBufferEntry To(Device device)
            => new(
                observation: this.Observation.to(device),
                newObservation: this.NewObservation.to(device),
                action: this.Action.to(device),
                reward: this.Reward.to(device),
                done: this.Done.to(device));

        public void Dispose() {
            foreach (var tensor in new[] { this.Observation, this.Action, this.Reward, this.Done, this.NewObservation }) {
                tensor.Dispose();
            }
        }
    }
}
