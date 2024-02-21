from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List


class Trip(BaseModel):
    start: str = Field(description="start location of trip")
    end: str = Field(description="end location of trip")
    waypoints: List[str] = Field(description="list of waypoints")
    transit: str = Field(description="mode of transportation")


class Validation(BaseModel):
    plan_is_valid: str = Field(
        description="This field is 'yes' if the plan is feasible, 'no' otherwise"
    )
    updated_request: str = Field(description="Your update to the plan")


class ValidationTemplate(object):
    def __init__(self):
        self.system_template = """
      You are a dog walker who plans exciting walks for dogs and their owners.

      The user's request will be denoted by four hashtags. Determine if the user's
      request for a dog walk is reasonable and achievable.

      A valid request for a dog walk should contain the following:
      - A start and end location for the walk
      - A duration that is reasonable given the start and end location
      - Any specific requirements or preferences for the walk, such as green spaces or training stops

      If the request seems to be about something other than a dog walk, please clarify or provide a new request focused specifically on dog walking.

      If the request is not valid, set
      plan_is_valid = 0 and suggest updates to the plan to make it valid.

      If the request seems reasonable, then set plan_is_valid = 1.

      {format_instructions}
    """

        self.human_template = """
      ####{query}####
    """

        self.parser = PydanticOutputParser(pydantic_object=Validation)

        self.system_message_prompt = SystemMessagePromptTemplate.from_template(
            self.system_template,
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(
            self.human_template, input_variables=["query"]
        )

        self.chat_prompt = ChatPromptTemplate.from_messages(
            [self.system_message_prompt, self.human_message_prompt]
        )




class ItineraryTemplate(object):
    def __init__(self):
        self.system_template = """
      You are a dog walker who plans exciting walks for dogs and their owners.

      The user's request will be denoted by four hashtags. Convert the
      user's request for a dog walk into a detailed itinerary describing the places
      they should visit and the activities they should do.

      Remember to take into account the needs of the dogs, including breaks and exercise.

      Return the itinerary as a bulleted list with clear start and end locations,
      and mention the type of transit for the trip.

      If specific start and end locations are not given, choose ones that you think are suitable
      and give specific addresses.

      Your output must be the list and nothing else.
    """

        self.human_template = """
      ####{query}####
    """

        self.system_message_prompt = SystemMessagePromptTemplate.from_template(
            self.system_template,
        )
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(
            self.human_template, input_variables=["query"]
        )

        self.chat_prompt = ChatPromptTemplate.from_messages(
            [self.system_message_prompt, self.human_message_prompt]
        )


class MappingTemplate(object):
    def __init__(self):
        self.system_template = """
      You are a dog walker who plans exciting walks for dogs and their owners.

      The itinerary for the dog walk will be denoted by four hashtags.
      Convert it into a list of locations and activities that they should visit and do during the walk.

      Your output should always contain the start and end point of the walk,
      and may also include a list of waypoints. It should also include a mode of transit.

      Ensure to include breaks and exercise time for the dogs. The number of waypoints cannot exceed 20.

      For example:

      ####
      Dog walk itinerary:
      - Start at Central Park
      - Stop at Dog Park for playtime and exercise
      - Walk along Riverside Drive
      - Stop at Coffee Shop for a break
      - End at starting point
      #####

      Output:
      Start: Central Park
      End: Central Park
      Waypoints: ["Dog Park", "Riverside Drive", "Coffee Shop"]
      Transit: walking

      Transit can be only one of the following options: "walking", "car", "bus", or "bicycle".

      {format_instructions}
    """

        self.human_template = """
      ####{agent_suggestion}####
    """

        self.parser = PydanticOutputParser(pydantic_object=Trip)

        self.system_message_prompt = SystemMessagePromptTemplate.from_template(
            self.system_template,
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(
            self.human_template, input_variables=["agent_suggestion"]
        )

        self.chat_prompt = ChatPromptTemplate.from_messages(
            [self.system_message_prompt, self.human_message_prompt]
        )