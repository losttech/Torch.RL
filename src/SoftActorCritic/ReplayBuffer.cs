namespace LostTech.Torch.RL {
    using System;
    using TorchSharp;
    using TorchSharp.Tensor;
    using static System.Linq.Enumerable;

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
                observation: Float32Tensor.zeros(new long[] { size, observationDimensions }),
                newObservation: Float32Tensor.zeros(new long[] { size, observationDimensions }),
                action: Float32Tensor.zeros(new long[] { size, actionDimensions }),
                reward: Float32Tensor.zeros(size),
                done: Float32Tensor.zeros(size)
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
            using var noGrad = new AutoGradMode(false);
            var indices = Int64Tensor.randint(max: this.Size, new long[] { batchSize });
            return new ReplayBufferEntry(
                observation: this.buffer.Observation[TorchTensorIndex.Tensor(indices)],
                newObservation: this.buffer.NewObservation[TorchTensorIndex.Tensor(indices)],
                action: this.buffer.Action[TorchTensorIndex.Tensor(indices)],
                reward: this.buffer.Reward[TorchTensorIndex.Tensor(indices)],
                done: this.buffer.Done[TorchTensorIndex.Tensor(indices)]
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

            using var noGrad = new AutoGradMode(false);

            if (this.Size == this.Capacity)
                this.ptr = Int32Tensor.randint(max: this.Capacity / this.batchSize, new long[] { 1 }).ToScalar().ToInt32() * this.batchSize;

            foreach (int batchElement in Range(0, this.batchSize)) {
                this.buffer.Observation[this.ptr + batchElement] = observation.Observation[batchElement];
                this.buffer.NewObservation[this.ptr + batchElement] = observation.NewObservation[batchElement];
                this.buffer.Action[this.ptr + batchElement] = observation.Action[batchElement];
                this.buffer.Reward[this.ptr + batchElement] = observation.Reward[batchElement];
                this.buffer.Done[this.ptr + batchElement] = observation.Done[batchElement];
            }
            this.ptr = (this.ptr + this.batchSize) % this.Capacity;
            this.Size = Math.Min(this.Size + this.batchSize, this.Capacity);
        }
        /// <summary>Current number of observations in the buffer</summary>
        public int Size { get; private set; }
        /// <summary>Buffer capacity for observations</summary>
        public int Capacity { get; }
    }
}
