"""
AI explainer module for CPU scheduling simulations.
Generates an academic, educational report analyzing simulator results using the Google Gemini API.
"""

import google.generativeai as genai

# --- Gemini Explainer Implementation ---

def get_explanation(
    metrics: dict[str, dict], 
    recommendations: dict[str, dict], 
    api_key: str
) -> str:
    """
    Constructs a detailed prompt from simulation metrics and recommendations,
    and calls Google Gemini API (gemini-1.5-flash) to generate an explanation report.

    Parameters:
        metrics (dict[str, dict]): Dictionary mapping algorithm names to their computed metrics.
        recommendations (dict[str, dict]): Rule-based recommendations for different scenarios.
        api_key (str): Google Gemini API Key.

    Returns:
        str: Generated explanation text in markdown format, or an error message.
    """
    if not api_key or api_key.strip() == "":
        return (
            "⚠️ **API Key Missing**: Please provide a valid Gemini API key in the panel "
            "or configure it in Streamlit secrets to generate the explanation."
        )

    # 1. Format the metrics for the prompt
    formatted_metrics = ""
    for algo, data in metrics.items():
        formatted_metrics += f"Algorithm: {algo}\n"
        formatted_metrics += f"  - Average Waiting Time: {data['avg_waiting_time']} time units\n"
        formatted_metrics += f"  - Average Turnaround Time: {data['avg_turnaround_time']} time units\n"
        formatted_metrics += f"  - Average Response Time: {data['avg_response_time']} time units\n"
        formatted_metrics += f"  - Throughput: {data['throughput']} processes/time unit\n"
        formatted_metrics += f"  - CPU Utilization: {data['cpu_utilization']}%\n\n"

    # 2. Format recommendations for the prompt
    formatted_recs = ""
    for scenario, rec in recommendations.items():
        formatted_recs += f"- Scenario: {scenario}\n"
        formatted_recs += f"  - Recommended Algorithm: {rec['algorithm']}\n"
        formatted_recs += f"  - Reason: {rec['reason']}\n\n"

    # 3. Build the prompt
    prompt = (
        "You are an Operating Systems expert and computer science professor.\n\n"
        "Given these CPU scheduling simulation results:\n"
        f"{formatted_metrics}"
        "And these recommendations:\n"
        f"{formatted_recs}"
        "Please explain in 4–6 paragraphs:\n"
        "1. Which algorithm performed best overall and why based on the numbers\n"
        "2. Which algorithm is best for fairness and why\n"
        "3. Which algorithm suits embedded or real-time systems and why\n"
        "4. Which algorithm maximizes throughput and what trade-offs it makes\n"
        "5. A brief guidance for a student choosing between these algorithms\n\n"
        "Be specific, reference the actual metric values, and keep the language clear for an undergraduate OS student.\n"
        "Use standard markdown (headings, bold text, bullet points) for structure, and maintain a professional yet engaging academic tone."
    )

    # 4. Invoke the API
    try:
        genai.configure(api_key=api_key.strip())
        
        # Using the gemini-1.5-flash model as specified
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.4
            )
        )
        
        if response and response.text:
            return response.text
        else:
            return "❌ **API Error**: No response was returned from the Gemini service."

    except Exception as e:
        return (
            f"❌ **API Error occurred**: {str(e)}\n\n"
            "Please check that your API key is valid, active, and that you have internet connectivity."
        )
