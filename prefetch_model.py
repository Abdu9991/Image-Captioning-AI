import os

from transformers import (
    AutoProcessor,
    BlipForConditionalGeneration,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    BlipProcessor,
)


def main():
    model_source = os.environ.get("FINETUNED_MODEL_PATH") or os.environ["MODEL_ID"]
    normalized = model_source.lower()

    if "qwen2.5-vl" in normalized or "qwen-vl" in normalized:
        AutoProcessor.from_pretrained(model_source)
        return

    if "blip2" in normalized:
        Blip2Processor.from_pretrained(model_source)
        Blip2ForConditionalGeneration.from_pretrained(
            model_source,
            low_cpu_mem_usage=True,
        )
        return

    BlipProcessor.from_pretrained(model_source)
    BlipForConditionalGeneration.from_pretrained(
        model_source,
        low_cpu_mem_usage=True,
    )


if __name__ == "__main__":
    main()