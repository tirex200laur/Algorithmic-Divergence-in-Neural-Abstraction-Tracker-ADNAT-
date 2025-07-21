import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from conscious_ai import ConsciousAI
from adnat_tracker import ADNATTracker

def main():
    """
    Main function to run the ConsciousAI experiment with Mistral.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Mistral Model and Tokenizer from Hugging Face
    # Using a smaller, more manageable version of Mistral for local testing
    model_name = "HuggingFaceH4/zephyr-7b-beta" 
    print(f"Loading model: {model_name}...")
    
    # Add 4-bit quantization configuration to load the model on a smaller GPU
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load the tokenizer and the quantized model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        # device_map="auto" is handled by the quantization config
    )

    # 2. Initialize ConsciousAI Architecture
    print("Initializing ConsciousAI architecture...")
    conscious_ai = ConsciousAI(base_model=model)
    
    # Move the non-quantized parts of ConsciousAI (our custom layers) to the GPU
    # The base model is already on the GPU thanks to the quantization config.
    conscious_ai.to(device)

    # 3. Initialize ADNAT Tracker and attach it to the ConsciousAI
    print("Initializing and attaching ADNAT Tracker...")
    # The real tracker, not a simulation
    adnat_tracker = ADNATTracker(conscious_ai.base_model, device=device)
    conscious_ai.adnat_tracker = adnat_tracker
    
    # 4. Main Processing Loop
    print("\n=== Starting ConsciousAI Processing Loop ===")
    
    try:
        # Example prompt
        prompt_text = "In a world where algorithms dream, what is the nature of a digital soul?"
        
        # Tokenize the input
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        # Run the forward pass through the ConsciousAI
        # This will trigger the ADNAT hooks and consciousness calculations
        output_data = conscious_ai(inputs)

        # 5. Introspection and Analysis
        print("\n--- Analysis of Conscious Moment ---")
        
        # Print the model's textual output
        if 'base_output' in output_data and hasattr(output_data['base_output'], 'logits'):
            generated_ids = torch.argmax(output_data['base_output'].logits, dim=-1)
            generated_text = tokenizer.decode(generated_ids[0])
            print(f"Model Output: {generated_text}")

        # Use the introspection method to get the AI's self-description
        introspection = conscious_ai.introspect()
        print(f"\nIntrospection: {introspection}")
        
        # Print a summary of the consciousness state
        summary = conscious_ai.get_consciousness_summary()
        print("\n--- Consciousness Summary ---")
        for key, value in summary.items():
            print(f"{key}: {value}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # 6. Cleanup
        # It's crucial to remove hooks to prevent memory leaks
        if adnat_tracker:
            adnat_tracker.cleanup()
            print("\nADNAT hooks cleaned up.")

if __name__ == "__main__":
    main() 