# Step 0: Azure OpenAI via LiteLLM (used by CrewAI)
import os
from crewai import Agent, Task, Crew, LLM

# ==== YOUR AZURE OPENAI SETTINGS ====
# NOTE: 'AZURE_DEPLOYMENT' must match the *deployment name* in Azure OpenAI (you said it's "gpt-4o")
os.environ["AZURE_API_BASE"] = "https://shace-mejiegxb-eastus2.cognitiveservices.azure.com/"
os.environ["AZURE_API_KEY"] = "9LTf4DOFBLDVHcCm9LaEcJ1hLUK3QIhzDVqGOhKqh6nsF3VBIvoLJQQJ99BHACHYHv6XJ3w3AAAAACOG77Gk"
os.environ["AZURE_API_VERSION"] = "2024-12-01-preview"
AZURE_DEPLOYMENT = "gpt-4o"  # deployment name in your Azure OpenAI resource

# Create a single LLM instance backed by Azure (via LiteLLM)
llm = LLM(
    model=f"azure/{AZURE_DEPLOYMENT}",  # provider/deployment_name
    api_version=os.environ["AZURE_API_VERSION"],
    # optional tunables:
    temperature=0.4,
    max_tokens=1200,
)

# Step 1: Import required classes (already imported above)
# from crewai import Agent, Task, Crew

# Step 2: Define the agents (all share the same Azure-backed LLM)
country_tourism_agent = Agent(
    role='Country Tourism Specialist',
    goal='Provide comprehensive information about tourism in a specific country',
    backstory=(
        'An experienced travel consultant with deep knowledge about global destinations, '
        'culture, landmarks, seasons, visa policies, and must-see places.'
    ),
    llm=llm,
    verbose=True
)

itinerary_agent = Agent(
    role='Travel Itinerary Planner',
    goal='Create a detailed and efficient travel plan for the customer',
    backstory=(
        'An expert in building travel itineraries with over 10 years of experience. Skilled in mapping routes, '
        'choosing transport modes, and balancing travel times for optimal enjoyment.'
    ),
    llm=llm,
    verbose=True
)

custom_tour_agent = Agent(
    role='Custom Tour Designer',
    goal='Personalize the tour plan according to customer needs and preferences',
    backstory=(
        'A creative tour planner who specializes in tailoring trips based on customer budgets, '
        'interests, travel styles, and special requests to deliver unforgettable experiences.'
    ),
    llm=llm,
    verbose=True
)

# Step 3: Define the tasks
tourism_task = Task(
    description=(
        "Research Japan's tourism landscape. Include top cities to visit, cultural highlights, "
        "natural wonders, travel seasons, entry visa requirements, and must-see attractions."
    ),
    expected_output="A detailed summary of Japan's tourism overview and top recommended locations.",
    agent=country_tourism_agent
)

itinerary_task = Task(
    description=(
        "Using the information from the Country Tourism Specialist, create a 10-day travel itinerary "
        "for Japan. Include travel days, intercity movement, recommended hotels, and must-visit sites each day."
    ),
    expected_output="A structured day-by-day travel plan across cities in Japan including travel logistics.",
    agent=itinerary_agent,
    context=[tourism_task]
)

customization_task = Task(
    description=(
        "Tailor the 10-day Japan itinerary to suit a customer interested in food experiences, "
        "a moderate pace (not rushed), and a mid-range budget. Include recommendations for unique local experiences."
    ),
    expected_output="A personalized tour plan for Japan with food tours, relaxed schedule, and mid-budget options.",
    agent=custom_tour_agent,
    context=[itinerary_task]
)

# Step 4: Create the crew
crew = Crew(
    agents=[country_tourism_agent, itinerary_agent, custom_tour_agent],
    tasks=[tourism_task, itinerary_task, customization_task],
    verbose=True
)

# Step 5: Kick off the workflow
if __name__ == "__main__":
    result = crew.kickoff()
    print("\n✅ Final Custom Tour Plan:\n")
    print(result)
