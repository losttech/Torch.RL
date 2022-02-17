namespace LostTech.Torch.RL.SoftActorCritic;

using System;

using TorchSharp;

using static System.Linq.Enumerable;
using static TorchSharp.torch;

/// <summary>
/// Circular buffer, that records agent observations.
/// </summary>
public class ReplayBuffer {
    readonly ReplayBufferEntry buffer;
    int ptr;
    readonly int batchSize;

    /// <summary>
    /// Creates new <see cref="ReplayBuffer"/>
    /// </summary>
    /// <param name="observationDimensions">Number of dimensions in observations.
    /// Each Observation assumed to be a n-element vector.</param>
    /// <param name="actionDimensions">Number of dimensions in actions.
    /// Each Action assumed to be a m-element vector.</param>
    /// <param name="size">Maximum number of records in the buffer.
    /// When this number is reached, older records get overwritten randomly</param>
    /// <param name="batchSize">Number of observations per time step (usually is the number of 
    /// agents)</param>
    public ReplayBuffer(int observationDimensions, int actionDimensions, int size, int batchSize) {
        this.buffer = new ReplayBufferEntry(
            observation: zeros(new long[] { size, observationDimensions }),
            newObservation: zeros(new long[] { size, observationDimensions }),
            action: zeros(new long[] { size, actionDimensions }),
            reward: zeros(size),
            done: zeros(size)
        );
        this.Capacity = size;
        this.batchSize = batchSize;
        if (size % batchSize != 0)
            throw new ArgumentException($"{nameof(size)} must be mutiplicative of {nameof(batchSize)}");
    }
    /// <summary>
    /// Pick random observations from the recorded history.
    /// </summary>
    /// <param name="batchSize">Number of observations to pick</param>
    public ReplayBufferEntry SampleBatch(int batchSize) {
        using var noGrad = no_grad();
        var indices = randint(high: this.Size, new long[] { batchSize }, dtype: ScalarType.Int64);
        var tensorIndices = TensorIndex.Tensor(indices);
        return new ReplayBufferEntry(
            observation: this.buffer.Observation[tensorIndices],
            newObservation: this.buffer.NewObservation[tensorIndices],
            action: this.buffer.Action[tensorIndices],
            reward: this.buffer.Reward[tensorIndices],
            done: this.buffer.Done[tensorIndices]
        );
    }
    /// <summary>
    /// Store new observation(s) to the buffer, overwriting random ones when necessary
    /// </summary>
    public void Store(ReplayBufferEntry observation) {
        if (observation.Observation.shape[0] != this.batchSize)
            throw new ArgumentException(
                message: "The first dimension of input must match batchSize",
                paramName: nameof(observation));

        using var noGrad = no_grad();

        if (this.Size == this.Capacity)
            this.ptr = randint(high: this.Capacity / this.batchSize, new long[] { 1 }).ToScalar().ToInt32() * this.batchSize;

        foreach (int batchElement in Range(0, this.batchSize)) {
            this.buffer.Observation[this.ptr + batchElement].copy_(observation.Observation[batchElement]);
            this.buffer.NewObservation[this.ptr + batchElement].copy_(observation.NewObservation[batchElement]);
            this.buffer.Action[this.ptr + batchElement].copy_(observation.Action[batchElement]);
            this.buffer.Reward[this.ptr + batchElement].copy_(observation.Reward[batchElement]);
            this.buffer.Done[this.ptr + batchElement].copy_(observation.Done[batchElement]);
        }
        this.ptr = (this.ptr + this.batchSize) % this.Capacity;
        this.Size = Math.Min(this.Size + this.batchSize, this.Capacity);
    }
    /// <summary>Current number of observations in the buffer</summary>
    public int Size { get; private set; }
    /// <summary>Buffer capacity for observations</summary>
    public int Capacity { get; }
}
