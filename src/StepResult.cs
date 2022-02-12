namespace LostTech.AI.RL;

public class StepResult<TReward, TObservation> {
    public TReward Reward { get; set; }
    public TObservation Observation { get; set; }
}
