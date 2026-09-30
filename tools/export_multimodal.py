"""Record verifiable Qwen2-VL image tokens and language attention.

Requires a complete local checkpoint. No synthetic fallback is produced.
Heatmaps show final language-layer attention averaged over heads and normalized
within image tokens; attention is a measurement, not a causal explanation.
"""

import argparse
import gzip
import hashlib
import json
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "public/multimodal"
SOURCES = pathlib.Path("D:/models/photos")
EXAMPLES = [
    ("cat", "橘猫", "这张照片里有什么？", "cat.jpg"),
    ("street", "街景", "描述这张照片的主要场景。", "street.jpg"),
    ("food", "小笼包", "这张照片里是什么食物？", "food.jpg"),
    ("sign", "路牌", "读出照片中左上方蓝色路牌上的主要地名。", "sign.jpg"),
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    tmp.write_bytes(raw)
    tmp.replace(path)
    if path.name != "manifest.json":
        compressed = path.with_suffix(path.suffix + ".gz")
        tmp_gzip = compressed.with_suffix(compressed.suffix + ".tmp")
        tmp_gzip.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
        tmp_gzip.replace(compressed)


def ensure_complete_model(path):
    index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    missing = [name for name in set(index["weight_map"].values()) if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"模型权重缺失: {', '.join(missing)}")


def local_model_revision(path):
    metadata = path / ".cache/huggingface/download/config.json.metadata"
    return metadata.read_text(encoding="utf-8").splitlines()[0] if metadata.is_file() else None


def record_one(model, processor, source, name, question, image_path, max_tokens):
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": question},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    grid = inputs.image_grid_thw[0].tolist()
    temporal, raw_rows, raw_cols = (int(x) for x in grid)
    merge = int(model.config.vision_config.spatial_merge_size)
    rows, cols = raw_rows // merge, raw_cols // merge
    image_positions = (inputs.input_ids[0] == model.config.image_token_id).nonzero().flatten().tolist()
    if temporal != 1 or len(image_positions) != rows * cols:
        raise ValueError(f"图像词元网格不匹配: grid={grid}, image tokens={len(image_positions)}")

    with torch.inference_mode():
        visual = model.visual(inputs.pixel_values.to(model.visual.dtype), grid_thw=inputs.image_grid_thw).float().cpu()
        if visual.shape[0] != len(image_positions):
            raise ValueError("视觉向量数与图像词元数不一致")
        generated = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                   output_attentions=False)
        token_ids = generated[0, inputs.input_ids.shape[1]:].tolist()
        eos = {model.config.eos_token_id, model.config.pad_token_id}
        token_ids = [token for token in token_ids if token not in eos]
        if not token_ids:
            raise ValueError("模型未生成可展示的回答词元")
        answer = processor.tokenizer.decode(token_ids, skip_special_tokens=True)

        # Query position predicting token i is prompt[-1] for i=0, else token i-1.
        full_ids = torch.cat([inputs.input_ids, torch.tensor([token_ids], device=model.device)], dim=1)
        forward_inputs = dict(inputs)
        forward_inputs["input_ids"] = full_ids
        forward_inputs["attention_mask"] = torch.ones_like(full_ids)
        forward_inputs.pop("pixel_values_videos", None)
        forward_inputs.pop("video_grid_thw", None)
        output = model(**forward_inputs, output_attentions=True, use_cache=False, return_dict=True)
        if not output.attentions or output.attentions[-1] is None:
            raise RuntimeError("未获得语言层注意力；请确认 attn_implementation='eager'")
        last = output.attentions[-1][0].float()
        prompt_len = inputs.input_ids.shape[1]
        records = []
        for idx, token_id in enumerate(token_ids):
            per_head = last[:, prompt_len + idx - 1, image_positions]
            weights = per_head.mean(dim=0)
            image_mass = float(weights.sum().item())
            if image_mass <= 0:
                raise ValueError(f"词元 {idx} 的图像注意力总量为零")
            head_masses = per_head.sum(dim=1)
            if torch.any(head_masses <= 0):
                raise ValueError(f"词元 {idx} 有注意力头对图像权重为零")
            records.append({
                "id": int(token_id),
                "text": processor.tokenizer.decode([token_id], skip_special_tokens=True),
                "imageAttentionMass": round(image_mass, 8),
                "weights": [round(float(w / image_mass), 7) for w in weights.cpu().tolist()],
                "heads": [{
                    "imageAttentionMass": round(float(head_masses[h]), 8),
                    "weights": [round(float(w / head_masses[h]), 7) for w in per_head[h].cpu().tolist()],
                } for h in range(per_head.shape[0])],
            })
        del output

    patches = [{
        "id": idx, "row": idx // cols, "col": idx % cols,
        "norm": round(float(vector.norm().item()), 5),
        "head": [round(float(x), 5) for x in vector[:12].tolist()],
    } for idx, vector in enumerate(visual)]
    return {
        "schema": 2, "source": "model", "id": source, "name": name,
        "question": question, "answer": answer,
        "image": f"images/{source}.jpg", "imageSha256": sha256(image_path),
        "originalSize": [image.width, image.height],
        "processedSize": [raw_cols * 14, raw_rows * 14],
        "grid": {"rows": rows, "cols": cols, "rawRows": raw_rows, "rawCols": raw_cols,
                 "rawPatchSize": int(model.config.vision_config.patch_size), "merge": merge,
                 "featureDim": int(visual.shape[1])},
        "attention": {"layer": int(model.config.num_hidden_layers) - 1,
                      "heads": int(model.config.num_attention_heads),
                      "method": "mean heads; normalized among image tokens only"},
        "patches": patches, "tokens": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=pathlib.Path, default=pathlib.Path("D:/models/Qwen2-VL-2B-Instruct"))
    parser.add_argument("--photos", type=pathlib.Path, default=SOURCES)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--only", choices=[item[0] for item in EXAMPLES], help="只处理一张照片")
    parser.add_argument("--probe", action="store_true", help="验证真实提取但不写入站点")
    args = parser.parse_args()
    if args.only and not args.probe:
        parser.error("--only 仅可与 --probe 一起使用，避免产生混合版本数据")
    ensure_complete_model(args.model)
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError("当前导出需要 CUDA GPU；未生成任何替代数据")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    # 256 merged image tokens keep eager full-sequence attention within 8 GB VRAM.
    processor.image_processor.max_pixels = 256 * 28 * 28
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    results = []
    captures = []
    for key, name, question, filename in EXAMPLES:
        if args.only and key != args.only:
            continue
        image_path = args.photos / filename
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        print(f"录制 {name}: {image_path}", flush=True)
        data = record_one(model, processor, key, name, question, image_path, args.max_new_tokens)
        captures.append((key, image_path, data))
        results.append({"id": key, "name": name, "question": question,
                        "answer": data["answer"], "image": data["image"],
                        "imageSha256": data["imageSha256"], "dataFile": f"{key}.json",
                        "grid": data["grid"]})
        print(f"  {data['grid']['rows']}×{data['grid']['cols']} image tokens; {len(data['tokens'])} output tokens", flush=True)
    if not args.probe:
        for key, image_path, data in captures:
            shutil.copy2(image_path, OUT / "images" / f"{key}.jpg")
            atomic_json(OUT / "data" / f"{key}.json", data)
        atomic_json(OUT / "data/manifest.json", {
            "schema": 2, "source": "model", "model": "Qwen2-VL-2B-Instruct",
            "modelRevision": local_model_revision(args.model),
            "method": "Greedy generation; merged visual encoder outputs; final language layer attention averaged over 12 heads",
            "examples": results,
        })


if __name__ == "__main__":
    main()
