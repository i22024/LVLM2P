
from PIL import Image


fewshot_prompt_action=[

    ################################################################# orange arrow : right wards ################################################################# 

    # 1st example - rightwards - rightwards
    {
        "role": 'user',
        "parts": [
        f"""
            Position relationship:
            Orange arrow is facing rightwards.
            Target location is to the rightwards from the orange arrow.

            Print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach Target location
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
            """
            Next action probability distribution :
            Turn left: 0.0
            Turn right: 0.0 
            Go Forward : 1.0

            Answer:
            [0.0, 0.0, 1.0]
            """
        ]
    },
    # 2nd example - rightwards - upwards
    {
        "role": 'user',
        "parts": [
        f"""
            Position relationship:
            Orange arrow is facing rightwards.
            Target location is to the upwards from the orange arrow.

            Print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach Target location
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
        """
            Next action probability distribution :
            Turn left: 1.0
            Turn right: 0.0 
            Go Forward : 0.0

            Answer:
            [1.0, 0.0, 0.0]
        """
        ]
    }
    ,

    # 3rd example - rightwards - leftwards
    {
        "role": 'user',
        "parts": [
        f"""
            Position relationship:
            Orange arrow is facing rightwards.
            Target location is to the leftwards from the orange arrow.

            Print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach Target location
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
        """
            Next action probability distribution :
            Turn left: 0.5
            Turn right: 0.5 
            Go Forward : 0.0

            Answer:
            [0.5, 0.5, 0.0]
        """
        ]
    }
    ,
    # 4th example - rightwards - downwards
    {
        "role": 'user',
        "parts": [
        f"""
            Position relationship:
            Orange arrow is facing rightwards.
            Target location is to the downwards from the orange arrow.

            Print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach Target location
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
        """
            Next action probability distribution :
            Turn left: 0.0
            Turn right: 1.0 
            Go Forward : 0.0

            Answer:
            [0.0, 1.0, 0.0]
        """
        ]
    }
    ,
    # 5th example - rightwards - rightwards + upwards
    {
        "role": 'user',
        "parts": [
        f"""
            Position relationship:
            Orange arrow is facing rightwards.
            Target location is to the rightwards and upwards from the orange arrow.

            Print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach Target location
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
        """
            Next action probability distribution :
            Turn left: 0.0
            Turn right: 0.0 
            Go Forward : 1.0

            Answer:
            [0.0, 0.0, 1.0]
        """
        ]
    }
        ,

    # 6th example - rightwards - rightwards + downwards
    {
        "role": 'user',
        "parts": [
        f"""
            Position relationship:
            Orange arrow is facing rightwards.
            Target location is to the rightwards and downwards from the orange arrow.

            Print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach Target location
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
        """
            Next action probability distribution :
            Turn left: 0.0
            Turn right: 0.0
            Go Forward : 1.0

            Answer:
            [0.0, 0.0, 1.0]
        """
        ]
    }
        ,
    # 7th example - rightwards - leftwards + upwards
    {
        "role": 'user',
        "parts": [
        f"""
            Position relationship:
            Orange arrow is facing rightwards.
            Target location is to the leftwards and upwards from the orange arrow.

            Print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach Target location
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
        """
            Next action probability distribution :
            Turn left: 1.0
            Turn right: 0.0 
            Go Forward : 0.0

            Answer:
            [1.0, 0.0, 0.0]
        """
        ]
    }
    ,
    # 8th example - rightwards - leftwards + downwards
    {
        "role": 'user',
        "parts": [
        f"""
            Position relationship:
            Orange arrow is facing rightwards.
            Target location is to the leftwards and downwards from the orange arrow.

            Print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach Target location
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
        """
            Next action probability distribution :
            Turn left: 0.0
            Turn right: 1.0 
            Go Forward : 0.0

            Answer:
            [0.0, 1.0, 0.0]
        """
        ]
    }
    ################################################################# orange arrow : left wards ################################################################# 
    ,
    # 9th example - leftwards - rightwards
    {
        "role": 'user',
        "parts": [
        f"""
            Position relationship:
            Orange arrow is facing leftwards.
            Target location is to the rightwards from the orange arrow.

            Print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach Target location
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
            """
            Next action probability distribution :
            Turn left: 0.5
            Turn right: 0.5 
            Go Forward : 0.0

            Answer:
            [0.5, 0.5, 0.0]
            """
        ]
    }
    ,
    # 10th example - leftwards - leftwards
    {
        "role": 'user',
        "parts": [
        f"""
            Position relationship:
            Orange arrow is facing leftwards.
            Target location is to the leftwards from the orange arrow.

            Print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach Target location
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
            """
            Next action probability distribution :
            Turn left: 0.0
            Turn right: 0.0 
            Go Forward : 1.0

            Answer:
            [0.0, 0.0, 1.0]
            """
        ]
    }
        ,
    # 11th example - leftwards - upwards
    {
        "role": 'user',
        "parts": [
        f"""
            Position relationship:
            Orange arrow is facing leftwards.
            Target location is to the upwards from the orange arrow.

            Print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach Target location
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
            """
            Next action probability distribution :
            Turn left: 0.0
            Turn right: 1.0 
            Go Forward : 0.0

            Answer:
            [0.0, 1.0, 0.0]
            """
        ]
    },
        # 12th example - leftwards - downwards
    {
        "role": 'user',
        "parts": [
        f"""
            Position relationship:
            Orange arrow is facing leftwards.
            Target location is to the downwards from the orange arrow.

            Print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach Target location
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
            """
            Next action probability distribution :
            Turn left: 1.0
            Turn right: 0.0 
            Go Forward : 0.0

            Answer:
            [1.0, 0.0, 0.0]
            """
        ]
    }
    ,


    # 13th example - leftwards - rightwards + upwards
    {
        "role": 'user',
        "parts": [
        f"""
            Position relationship:
            Orange arrow is facing leftwards.
            Target location is to the rightwards and upwards from the orange arrow.

            Print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach Target location
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
        """
            Next action probability distribution :
            Turn left: 0.0
            Turn right: 1.0 
            Go Forward : 0.0

            Answer:
            [0.0, 1.0, 0.0]
        """
        ]
    }
    ,
    # 14th example - leftwards - rightwards + downwards
    {
        "role": 'user',
        "parts": [
        f"""
            Position relationship:
            Orange arrow is facing leftwards.
            Target location is to the rightwards and downwards from the orange arrow.

            Print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach Target location
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
        """
            Next action probability distribution :
            Turn left: 1.0
            Turn right: 0.0 
            Go Forward : 0.0

            Answer:
            [1.0, 0.0, 0.0]
        """
        ]
    }
    ,
    # 15th example - leftwards - leftwards + upwards
    {
        "role": 'user',
        "parts": [
        f"""
            Position relationship:
            Orange arrow is facing leftwards.
            Target location is to the leftwards and upwards from the orange arrow.

            Print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach Target location
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
        """
            Next action probability distribution :
            Turn left: 0.0
            Turn right: 0.0
            Go Forward : 1.0

            Answer:
            [0.0, 0.0, 1.0]
        """
        ]
    }
    ,
    # 16th example - leftwards - leftwards + downwards
    {
        "role": 'user',
        "parts": [
        f"""
            Position relationship:
            Orange arrow is facing leftwards.
            Target location is to the leftwards and downwards from the orange arrow.

            Print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach Target location
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
        """
            Next action probability distribution :
            Turn left: 0.0
            Turn right: 0.0 
            Go Forward : 0.0

            Answer:
            [0.0, 0.0, 1.0]
        """
        ]
    }
    

    ################################################################# orange arrow : upwards ################################################################# 
    ,
    # 17th example - upwards - rightwards
    {
        "role": 'user',
        "parts": [
        f"""
            Position relationship:
            Orange arrow is facing upwards.
            Target location is to the rightwards from the orange arrow.

            Print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach Target location
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
            """
            Next action probability distribution :
            Turn left: 0.0
            Turn right: 1.0 
            Go Forward : 0.0

            Answer:
            [0.0, 1.0, 0.0]
            """
        ]
    }
    ,
    # 18th example - upwards - leftwards
    {
        "role": 'user',
        "parts": [
        f"""
            Position relationship:
            Orange arrow is facing upwards.
            Target location is to the leftwards from the orange arrow.

            Print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach Target location
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
            """
            Next action probability distribution :
            Turn left: 1.0
            Turn right: 0.0 
            Go Forward : 0.0

            Answer:
            [1.0, 0.0, 0.0]
            """
        ]
    }
        ,
    # 19th example - upwards - upwards
    {
        "role": 'user',
        "parts": [
        f"""
            Position relationship:
            Orange arrow is facing upwards.
            Target location is to the upwards from the orange arrow.

            Print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach Target location
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
            """
            Next action probability distribution :
            Turn left: 0.0
            Turn right: 0.0 
            Go Forward : 1.0

            Answer:
            [0.0, 0.0, 1.0]
            """
        ]
    },
    # 20th example - upwards - downwards
    {
        "role": 'user',
        "parts": [
        f"""
            Position relationship:
            Orange arrow is facing upwards.
            Target location is to the downwards from the orange arrow.

            Print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach Target location
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
            """
            Next action probability distribution :
            Turn left: 0.5
            Turn right: 0.5 
            Go Forward : 0.0

            Answer:
            [0.5, 0.5, 0.0]
            """
        ]
    }
    ,


    # 21th example - upwards - rightwards + upwards
    {
        "role": 'user',
        "parts": [
        f"""
            Position relationship:
            Orange arrow is facing upwards.
            Target location is to the rightwards and upwards from the orange arrow.

            Print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach Target location
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
        """
            Next action probability distribution :
            Turn left: 0.0
            Turn right: 0.0
            Go Forward : 1.0

            Answer:
            [0.0, 0.0, 1.0]
        """
        ]
    }
    ,
    # 22th example - upwards - rightwards + downwards
    {
        "role": 'user',
        "parts": [
        f"""
            Position relationship:
            Orange arrow is facing upwards.
            Target location is to the rightwards and downwards from the orange arrow.

            Print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach Target location
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
        """
            Next action probability distribution :
            Turn left: 0.0
            Turn right: 1.0 
            Go Forward : 0.0

            Answer:
            [0.0, 1.0, 0.0]
        """
        ]
    }
    ,
    # 23th example - upwards - leftwards + upwards
    {
        "role": 'user',
        "parts": [
        f"""
            Position relationship:
            Orange arrow is facing upwards.
            Target location is to the leftwards and upwards from the orange arrow.

            Print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach Target location
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
        """
            Next action probability distribution :
            Turn left: 0.0
            Turn right: 0.0 
            Go Forward : 1.0

            Answer:
            [0.0, 0.0, 1.0]
        """
        ]
    }
    ,
    # 24th example - upwards - leftwards + downwards
    {
        "role": 'user',
        "parts": [
        f"""
            Position relationship:
            Orange arrow is facing upwards.
            Target location is to the leftwards and downwards from the orange arrow.

            Print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach Target location
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
        """
            Next action probability distribution :
            Turn left: 1.0
            Turn right: 0.0 
            Go Forward : 0.0

            Answer:
            [1.0, 0.0, 0.0]
        """
        ]
    }

    ################################################################# orange arrow : downwards ################################################################# 
    ,
    # 25th example - downwards - rightwards
    {
        "role": 'user',
        "parts": [
        f"""
            Position relationship:
            Orange arrow is facing downwards.
            Target location is to the rightwards from the orange arrow.

            Print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach Target location
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
            """
            Next action probability distribution :
            Turn left: 1.0
            Turn right: 0.0 
            Go Forward : 0.0

            Answer:
            [1.0, 0.0, 0.0]
            """
        ]
    }
    ,
    # 26th example - downwards - leftwards
    {
        "role": 'user',
        "parts": [
        f"""
            Position relationship:
            Orange arrow is facing downwards.
            Target location is to the leftwards from the orange arrow.

            Print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach Target location
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
            """
            Next action probability distribution :
            Turn left: 0.0
            Turn right: 1.0 
            Go Forward : 0.0

            Answer:
            [0.0, 1.0, 0.0]
            """
        ]
    }
        ,
    # 27th example - downwards - upwards
    {
        "role": 'user',
        "parts": [
        f"""
            Position relationship:
            Orange arrow is facing downwards.
            Target location is to the upwards from the orange arrow.

            Print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach Target location
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
            """
            Next action probability distribution :
            Turn left: 0.5
            Turn right: 0.5 
            Go Forward : 0.0

            Answer:
            [0.5, 0.5, 0.0]
            """
        ]
    },
    # 28th example - downwards - downwards
    {
        "role": 'user',
        "parts": [
        f"""
            Position relationship:
            Orange arrow is facing downwards.
            Target location is to the downwards from the orange arrow.

            Print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach Target location
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
            """
            Next action probability distribution :
            Turn left: 0.0
            Turn right: 0.0 
            Go Forward : 1.0

            Answer:
            [0.0, 0.0, 1.0]
            """
        ]
    }
    ,


    # 29th example - downwards - rightwards + upwards
    {
        "role": 'user',
        "parts": [
        f"""
            Position relationship:
            Orange arrow is facing downwards.
            Target location is to the rightwards and upwards from the orange arrow.

            Print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach Target location
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
        """
            Next action probability distribution :
            Turn left: 1.0
            Turn right: 0.0
            Go Forward : 0.0

            Answer:
            [1.0, 0.0, 0.0]
        """
        ]
    }
    ,
    # 30th example - downwards - rightwards + downwards
    {
        "role": 'user',
        "parts": [
        f"""
            Position relationship:
            Orange arrow is facing downwards.
            Target location is to the rightwards and downwards from the orange arrow.

            Print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach Target location
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
        """
            Next action probability distribution :
            Turn left: 0.0
            Turn right: 0.0 
            Go Forward : 1.0

            Answer:
            [0.0, 0.0, 1.0]
        """
        ]
    }
    ,
    # 31th example - downwards - leftwards + upwards
    {
        "role": 'user',
        "parts": [
        f"""
            Position relationship:
            Orange arrow is facing downwards.
            Target location is to the leftwards and upwards from the orange arrow.

            Print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach Target location
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
        """
            Next action probability distribution :
            Turn left: 0.0
            Turn right: 1.0 
            Go Forward : 0.0

            Answer:
            [0.0, 1.0, 0.0]
        """
        ]
    }
    ,
    # 32th example - downwards - leftwards + downwards
    {
        "role": 'user',
        "parts": [
        f"""
            Position relationship:
            Orange arrow is facing downwards.
            Target location is to the leftwards and downwards from the orange arrow.

            Print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach Target location
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
        """
            Next action probability distribution :
            Turn left: 0.0
            Turn right: 0.0 
            Go Forward : 1.0

            Answer:
            [0.0, 0.0, 1.0]
        """
        ]
    }


]

