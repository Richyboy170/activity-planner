import os
import shutil
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel
import onnxruntime as ort
import numpy as np

def convert_to_onnx(model_name='sentence-transformers/all-MiniLM-L6-v2', output_dir='models/onnx'):
    """
    Convert a SentenceTransformer model to ONNX format.
    """
    print(f"Converting {model_name} to ONNX...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    # Create dummy input
    dummy_input = tokenizer("This is a sample sentence", return_tensors="pt")
    
    # Export to ONNX
    onnx_path = output_path / "model.onnx"
    
    torch.onnx.export(
        model,
        (dummy_input['input_ids'], dummy_input['attention_mask'], dummy_input['token_type_ids']),
        str(onnx_path),
        input_names=['input_ids', 'attention_mask', 'token_type_ids'],
        output_names=['last_hidden_state', 'pooler_output'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'token_type_ids': {0: 'batch_size', 1: 'sequence_length'},
            'last_hidden_state': {0: 'batch_size', 1: 'sequence_length'},
            'pooler_output': {0: 'batch_size'}
        },
        opset_version=14
    )
    
    print(f"Model saved to {onnx_path}")
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer saved to {output_dir}")
    
    # Verify ONNX model
    print("Verifying ONNX model...")
    ort_session = ort.InferenceSession(str(onnx_path))
    
    onnx_inputs = {
        'input_ids': dummy_input['input_ids'].numpy(),
        'attention_mask': dummy_input['attention_mask'].numpy(),
        'token_type_ids': dummy_input['token_type_ids'].numpy()
    }
    
    onnx_outputs = ort_session.run(None, onnx_inputs)
    
    print("Verification successful!")
    print(f"Output shape: {onnx_outputs[0].shape}")

    # Quantize model (Optional but recommended for size)
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantized_model_path = output_path / "model_quantized.onnx"
        print("Quantizing model...")
        quantize_dynamic(
            str(onnx_path),
            str(quantized_model_path),
            weight_type=QuantType.QUInt8
        )
        print(f"Quantized model saved to {quantized_model_path}")
        
        # Check sizes
        orig_size = os.path.getsize(onnx_path) / (1024 * 1024)
        quant_size = os.path.getsize(quantized_model_path) / (1024 * 1024)
        print(f"Original size: {orig_size:.2f} MB")
        print(f"Quantized size: {quant_size:.2f} MB")
        
    except ImportError:
        print("onnxruntime-tools not installed, skipping quantization")
    except Exception as e:
        print(f"Quantization failed: {e}")

if __name__ == "__main__":
    convert_to_onnx()
