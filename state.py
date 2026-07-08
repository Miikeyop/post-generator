

# using TYpedDict
#most basic and widely used

from typing import TYpedDict


class state(TYpedDict):
    topic:str
    summary:str
    score:int


# using pydntic 

#pydantic is good for evalution 

from pydantic import BaseModel, field_validator

class state(BaseModel):
    topic:str
    summary:str
    score:int

    @field_validator("score")
    def validator(cls,v):
        if (v<0):
            raise ValueError("Score must be possitive")

#using dataclass
#rarely used

from dataclasses import dataclass,field

@dataclass
class state:
    topic:str=""
    summary:str=""
    message:list=field(default_factory=list)


from langgraph.graph import MessagesState

class state(MessagesState):
    user_name:str
    language:str
    
