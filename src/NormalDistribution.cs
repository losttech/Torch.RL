namespace LostTech.Torch.RL {
    using System;
    using System.Linq;
    using TorchSharp;
    using TorchSharp.Tensor;

    class NormalDistribution {
        public TorchTensor Mean { get; }
        public TorchTensor StdDev { get; }
        public NormalDistribution(TorchTensor mean, TorchTensor stddev) {
            this.Mean = mean ?? throw new ArgumentNullException(nameof(mean));
            this.StdDev = stddev ?? throw new ArgumentNullException(nameof(stddev));
            if (!this.Mean.shape.SequenceEqual(this.StdDev.shape))
                throw new ArgumentException(nameof(mean) + " and " + nameof(stddev) + " must have the same shape");
        }
        public TorchTensor Sample(long[]? shape = null, Device? device = null) {
            shape ??= this.Mean.shape;
            var eps = Float32Tensor.randn(shape, device);
            return eps * this.StdDev + this.Mean;
        }

        public TorchTensor LogProb(TorchTensor sample) {
            var variance = this.StdDev.pow(2);
            var logScale = this.StdDev.log();
            return -((sample - this.Mean).pow(2) / (2 * variance) - logScale - Math.Log(Math.Sqrt(2 * Math.PI)));
        }
    }
}
