namespace LostTech.Torch.RL;

using System;
using System.Linq;

using static TorchSharp.torch;

class NormalDistribution {
    public Tensor Mean { get; }
    public Tensor StdDev { get; }
    public NormalDistribution(Tensor mean, Tensor stddev) {
        this.Mean = mean ?? throw new ArgumentNullException(nameof(mean));
        this.StdDev = stddev ?? throw new ArgumentNullException(nameof(stddev));
        if (!this.Mean.shape.SequenceEqual(this.StdDev.shape))
            throw new ArgumentException(nameof(mean) + " and " + nameof(stddev) + " must have the same shape");
    }
    public Tensor Sample(long[]? shape = null, Device? device = null) {
        shape ??= this.Mean.shape;
        var eps = randn(shape, device: device);
        return eps * this.StdDev + this.Mean;
    }

    public Tensor LogProb(Tensor sample) {
        var variance = this.StdDev.pow(2);
        var logScale = this.StdDev.log();
        return -((sample - this.Mean).pow(2) / (2 * variance) - logScale - Math.Log(Math.Sqrt(2 * Math.PI)));
    }
}
