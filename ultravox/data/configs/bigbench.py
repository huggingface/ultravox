from ultravox.data import types

BB_BASE_CONFIG = types.DatasetConfig(
    name="big-bench-audio",
    path="eustlb/big-bench-audio-less-than-30s",
    splits=[types.DatasetSplitConfig(name="validation", num_samples=792)],
    assistant_template="{{official_answer}}",
    transcript_template="{{original_question}}",
)

configs = [
    BB_BASE_CONFIG,
]