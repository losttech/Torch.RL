namespace LostTech.AI.RL;

using System;
using System.Collections.Generic;
using System.Linq;

class RepeatObservationEnvironment : IEnvironment<StepResult<float[], float[]>, float[]> {
    readonly Random random = new();
    readonly float[] observations;
    readonly float[] rewards;
    public RepeatObservationEnvironment(int agentCount) {
        this.AgentCount = agentCount;
        this.observations = new float[agentCount];
        this.rewards = new float[agentCount];
    }
    public int AgentCount { get; }
    public int ObservationSize => 1;
    public int ActionSize => 1;
    public float ActionMin => -1;
    public float ActionMax => +1;

    public StepResult<float[], float[]> GetStepResult(string? agentGroupName) {
        if (agentGroupName is not null)
            throw new KeyNotFoundException();
        return new StepResult<float[], float[]> {
            Reward = (float[])this.rewards.Clone(),
            Observation = (float[])this.observations.Clone(),
        };
    }

    public void Reset() => this.Step();

    public void SetActions(string? agentGroupName, float[] actions) {
        if (agentGroupName is not null)
            throw new KeyNotFoundException();

        for (int i = 0; i < this.AgentCount; i++) {
            float action = Clip(actions[i], min: 0, max: 1);
            this.rewards[i] = 1 - Math.Abs(action - this.observations[i]);
        }
    }

    static float Clip(float v, float min, float max)
        => MathF.Min(max, MathF.Max(min, v));

    public void Step() {
        this.Fill(this.observations);
    }

    void Fill(float[] array) {
        for (int i = 0; i < array.Length; i++)
            array[i] = (float)this.random.NextDouble();
    }

    public float[] SampleAction() {
        float[] result = new float[this.AgentCount];
        this.Fill(result);
        return result;
    }
}
