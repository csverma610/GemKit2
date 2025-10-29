#!/usr/bin/env python3
"""
Test script demonstrating the refactored GeminiClient with GeminiConfig.
"""

from gemini_text_client import GeminiClient, GeminiConfig, ModelInput

# Example 1: Using default configuration
print("Example 1: Default Configuration")
print("-" * 50)
client1 = GeminiClient()
print(f"Model: {client1.model_name}")
print(f"Max Retries: {client1.max_retries}")
print(f"Timeout: {client1.timeout}s")
print()

# Example 2: Custom configuration
print("Example 2: Custom Configuration")
print("-" * 50)
config = GeminiConfig(
    model_name='gemini-2.5-pro',
    max_retries=5,
    timeout=180.0,
    rate_limit_calls=30,
    rate_limit_period=60.0
)
client2 = GeminiClient(config)
print(f"Model: {client2.model_name}")
print(f"Max Retries: {client2.max_retries}")
print(f"Timeout: {client2.timeout}s")
print(f"Rate Limit: {client2.rate_limit_calls} calls/{client2.rate_limit_period}s")
print()

# Example 3: Creating ModelInput (user input is separate from config)
print("Example 3: ModelInput (User Input)")
print("-" * 50)
model_input = ModelInput(
    user_prompt="What is the capital of France?",
    sys_prompt="You are a helpful geography tutor.",
    temperature=0.7,
    max_output_tokens=100
)
print(f"User Prompt: {model_input.user_prompt}")
print(f"System Prompt: {model_input.sys_prompt}")
print(f"Temperature: {model_input.temperature}")
print()

print("✓ All examples completed successfully!")
