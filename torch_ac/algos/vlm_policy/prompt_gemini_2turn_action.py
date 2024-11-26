
from PIL import Image

fewshot_prompt_action=[
    # 1st example
    {
        "role": 'user',
        "parts": [
        f"""
            You must follow these Orange arrow rules:
            'Go forward' moves forward one space in the direction the orange arrow is pointing.
            If the Orange arrow is pointing upwards and 'Turn left', the Orange arrow will point leftwards.
            If the Orange arrow is pointing upwards and 'Turn right', the Orange arrow will point rightwards.
            If the Orange arrow is pointing downwards and 'Turn left', the Orange arrow will point rightwards.
            If the Orange arrow is pointing downwards and 'Turn right', the Orange arrow will point leftwards.
            If the Orange arrow is pointing leftwards and 'Turn left', the Orange arrow will point downwards.
            If the Orange arrow is pointing leftwards and 'Turn right', the Orange arrow will point upwards.
            If the Orange arrow is pointing rightwards and 'Turn left', the Orange arrow will point upwards.
            If the Orange arrow is pointing rightwards and 'Turn right', the Orange arrow will point downwards.

            Current State:
            Based on top left (1,1),
            Orange arrow's current location : row 2 , column 2
            Orange arrow's current pointing : rightwards
            Target location's current location : row 2, column 4

            Problem :  Is the orange arrow currently at the target location?
            If so, print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 so that the orange arrow is pointing to row 1 column 4.
            If not, print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach row 2 column 4.
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
            """
            The orange arrow is not currently at the target location.

            To reach the target location (row 2, column 4), the orange arrow needs to Go forward.

            Action probability distribution:
            ['Turn left': 0.0, 'Turn right': 0.0, 'Go Forward': 1.0]

            [0.0, 0.0, 1.0]
            """
        ]
    },
    # 2nd example
    {
        "role": 'user',
        "parts": [
        f"""
            You must follow these Orange arrow rules:
            'Go forward' moves forward one space in the direction the orange arrow is pointing.
            If the Orange arrow is pointing upwards and 'Turn left', the Orange arrow will point leftwards.
            If the Orange arrow is pointing upwards and 'Turn right', the Orange arrow will point rightwards.
            If the Orange arrow is pointing downwards and 'Turn left', the Orange arrow will point rightwards.
            If the Orange arrow is pointing downwards and 'Turn right', the Orange arrow will point leftwards.
            If the Orange arrow is pointing leftwards and 'Turn left', the Orange arrow will point downwards.
            If the Orange arrow is pointing leftwards and 'Turn right', the Orange arrow will point upwards.
            If the Orange arrow is pointing rightwards and 'Turn left', the Orange arrow will point upwards.
            If the Orange arrow is pointing rightwards and 'Turn right', the Orange arrow will point downwards.

            Current State:
            Based on top left (1,1),
            Orange arrow's current location : row 2 , column 3
            Orange arrow's current pointing : rightwards
            Target location's current location : row 2, column 4

            Problem : Is the orange arrow currently at the target location?
            If so, print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 so that the orange arrow is pointing to row 1 column 4.
            If not, print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach row 2 column 4.
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
        """
        The orange arrow is not currently at the target location.

        To reach the target location (row 2, column 4), the orange arrow needs to Go forward.

        Action probability distribution:
        ['Turn left': 0.0, 'Turn right': 0.0, 'Go Forward': 1.0]

        [0.0, 0.0, 1.0] 
        """
        ]
    },
    # 3rd example
    {
        "role": 'user',
        "parts": [
        f"""
            You must follow these Orange arrow rules:
            'Go forward' moves forward one space in the direction the orange arrow is pointing.
            If the Orange arrow is pointing upwards and 'Turn left', the Orange arrow will point leftwards.
            If the Orange arrow is pointing upwards and 'Turn right', the Orange arrow will point rightwards.
            If the Orange arrow is pointing downwards and 'Turn left', the Orange arrow will point rightwards.
            If the Orange arrow is pointing downwards and 'Turn right', the Orange arrow will point leftwards.
            If the Orange arrow is pointing leftwards and 'Turn left', the Orange arrow will point downwards.
            If the Orange arrow is pointing leftwards and 'Turn right', the Orange arrow will point upwards.
            If the Orange arrow is pointing rightwards and 'Turn left', the Orange arrow will point upwards.
            If the Orange arrow is pointing rightwards and 'Turn right', the Orange arrow will point downwards.
            
            Current State:
            Based on top left (1,1),
            Orange arrow's current location : row 2 , column 4
            Orange arrow's current pointing : rightwards
            Target location's current location : row 2, column 4

            Problem : Is the orange arrow currently at the target location?
            If so, print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 so that the orange arrow is pointing to row 1 column 4.
            If not, print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach row 2 column 4.
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
            """
            The orange arrow is currently at the target location.

            To reach the target location (row 1, column 4), the orange arrow needs to turn left.

            Action probability distribution:
            ['Turn left': 1.0, 'Turn right': 0.0, 'Go Forward': 0.0]

            [1.0, 0.0, 0.0]
            """
        ]
    },

    # 4th example
    {
        "role": 'user',
        "parts": [
        f"""
            You must follow these Orange arrow rules:
            'Go forward' moves forward one space in the direction the orange arrow is pointing.
            If the Orange arrow is pointing upwards and 'Turn left', the Orange arrow will point leftwards.
            If the Orange arrow is pointing upwards and 'Turn right', the Orange arrow will point rightwards.
            If the Orange arrow is pointing downwards and 'Turn left', the Orange arrow will point rightwards.
            If the Orange arrow is pointing downwards and 'Turn right', the Orange arrow will point leftwards.
            If the Orange arrow is pointing leftwards and 'Turn left', the Orange arrow will point downwards.
            If the Orange arrow is pointing leftwards and 'Turn right', the Orange arrow will point upwards.
            If the Orange arrow is pointing rightwards and 'Turn left', the Orange arrow will point upwards.
            If the Orange arrow is pointing rightwards and 'Turn right', the Orange arrow will point downwards.

            Current State:
            Based on top left (1,1),
            Orange arrow's current location : row 2 , column 2
            Orange arrow's current pointing : downwards
            Target location's current location : row 2, column 4

            Problem : Is the orange arrow currently at the target location?
            If so, print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 so that the orange arrow is pointing to row 1 column 4.
            If not, print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach row 2 column 4.
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [  
            """
            The orange arrow is not currently at the target location.

            To reach the target location (row 2, column 4), the orange arrow needs to turn left.

            Action probability distribution:
            ['Turn left': 1.0, 'Turn right': 0.0, 'Go Forward': 0.0]

            [1.0, 0.0, 0.0] 
            """
        ]
    },
        # 5th example
    {
        "role": 'user',
        "parts": [
        f"""
            You must follow these Orange arrow rules:
            'Go forward' moves forward one space in the direction the orange arrow is pointing.
            If the Orange arrow is pointing upwards and 'Turn left', the Orange arrow will point leftwards.
            If the Orange arrow is pointing upwards and 'Turn right', the Orange arrow will point rightwards.
            If the Orange arrow is pointing downwards and 'Turn left', the Orange arrow will point rightwards.
            If the Orange arrow is pointing downwards and 'Turn right', the Orange arrow will point leftwards.
            If the Orange arrow is pointing leftwards and 'Turn left', the Orange arrow will point downwards.
            If the Orange arrow is pointing leftwards and 'Turn right', the Orange arrow will point upwards.
            If the Orange arrow is pointing rightwards and 'Turn left', the Orange arrow will point upwards.
            If the Orange arrow is pointing rightwards and 'Turn right', the Orange arrow will point downwards.

            Current State:
            Based on top left (1,1),
            Orange arrow's current location : row 5 , column 3
            Orange arrow's current pointing : leftwards
            Target location's current location : row 4, column 1

            Problem : Is the orange arrow currently at the target location?
            If so, print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 so that the orange arrow is pointing to row 4 column 1.
            If not, print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach row 4 column 2.
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
            """
            The orange arrow is not currently at the target location.

            To reach the target location (row 4, column 2), the orange arrow needs to Go forward or turn right.

            Action probability distribution:
            ['Turn left': 0.0, 'Turn right': 0.5, 'Go Forward': 0.5]

            [0.0, 0.5, 0.5] 
            """
        ]
    },
    
        # 6th example
    {
        "role": 'user',
        "parts": [
        f"""
            You must follow these Orange arrow rules:
            'Go forward' moves forward one space in the direction the orange arrow is pointing.
            If the Orange arrow is pointing upwards and 'Turn left', the Orange arrow will point leftwards.
            If the Orange arrow is pointing upwards and 'Turn right', the Orange arrow will point rightwards.
            If the Orange arrow is pointing downwards and 'Turn left', the Orange arrow will point rightwards.
            If the Orange arrow is pointing downwards and 'Turn right', the Orange arrow will point leftwards.
            If the Orange arrow is pointing leftwards and 'Turn left', the Orange arrow will point downwards.
            If the Orange arrow is pointing leftwards and 'Turn right', the Orange arrow will point upwards.
            If the Orange arrow is pointing rightwards and 'Turn left', the Orange arrow will point upwards.
            If the Orange arrow is pointing rightwards and 'Turn right', the Orange arrow will point downwards.

            Current State:
            Based on top left (1,1),
            Orange arrow's current location : row 3 , column 3
            Orange arrow's current pointing : uptwards
            Target location's current location : row 4, column 5

            Problem : Is the orange arrow currently at the target location?
            If so, print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 so that the orange arrow is pointing to row 4 column 6.
            If not, print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach row 4 column 5.
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
            """
            The orange arrow is not currently at the target location.

            To reach the target location (row 4, column 5), the orange arrow needs to turn right.

            Action probability distribution:
            ['Turn left': 0.0, 'Turn right': 1.0, 'Go Forward': 0.0]

            [0.0, 1.0, 0.0] 
            """
        ]
    },
    # 7th example
    {
        "role": 'user',
        "parts": [ 
        f"""
            You must follow these Orange arrow rules:
            'Go forward' moves forward one space in the direction the orange arrow is pointing.
            If the Orange arrow is pointing upwards and 'Turn left', the Orange arrow will point leftwards.
            If the Orange arrow is pointing upwards and 'Turn right', the Orange arrow will point rightwards.
            If the Orange arrow is pointing downwards and 'Turn left', the Orange arrow will point rightwards.
            If the Orange arrow is pointing downwards and 'Turn right', the Orange arrow will point leftwards.
            If the Orange arrow is pointing leftwards and 'Turn left', the Orange arrow will point downwards.
            If the Orange arrow is pointing leftwards and 'Turn right', the Orange arrow will point upwards.
            If the Orange arrow is pointing rightwards and 'Turn left', the Orange arrow will point upwards.
            If the Orange arrow is pointing rightwards and 'Turn right', the Orange arrow will point downwards.

            Current State:
            Based on top left (1,1),
            Orange arrow's current location : row 5 , column 4
            Orange arrow's current pointing : uptwards
            Target location's current location : row 5, column 4

            Problem : Is the orange arrow currently at the target location?
            If so, print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 so that the orange arrow is pointing to row 6 column 4.
            If not, print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach row 5 column 4.
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
            """
            The orange arrow is currently at the target location.

            To reach the target location (row 6, column 4), the orange arrow needs to turn left or turn right.

            Action probability distribution:
            ['Turn left': 0.5, 'Turn right': 0.5, 'Go Forward': 0.0]

            [0.5, 0.5, 0.0] 
            """
        ]
    },


        # 8th example
    {
        "role": 'user',
        "parts": [
        f"""
            You must follow these Orange arrow rules:
            'Go forward' moves forward one space in the direction the orange arrow is pointing.
            If the Orange arrow is pointing upwards and 'Turn left', the Orange arrow will point leftwards.
            If the Orange arrow is pointing upwards and 'Turn right', the Orange arrow will point rightwards.
            If the Orange arrow is pointing downwards and 'Turn left', the Orange arrow will point rightwards.
            If the Orange arrow is pointing downwards and 'Turn right', the Orange arrow will point leftwards.
            If the Orange arrow is pointing leftwards and 'Turn left', the Orange arrow will point downwards.
            If the Orange arrow is pointing leftwards and 'Turn right', the Orange arrow will point upwards.
            If the Orange arrow is pointing rightwards and 'Turn left', the Orange arrow will point upwards.
            If the Orange arrow is pointing rightwards and 'Turn right', the Orange arrow will point downwards.

            Current State:
            Based on top left (1,1),
            Orange arrow's current location : row 4 , column 5
            Orange arrow's current pointing : downwards
            Target location's current location : row 2, column 5

            Problem : Is the orange arrow currently at the target location?
            If so, print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 so that the orange arrow is pointing to row 1 column 5.
            If not, print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach row 2 column 5.
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
            """
            The orange arrow is not currently at the target location.

            To reach the target location (row 2, column 5), the orange arrow needs to turn left or turn right.

            Action probability distribution:
            ['Turn left': 0.5, 'Turn right': 0.5, 'Go Forward': 0.0]

            [0.5, 0.5, 0.0] 
            """
        ]
    },


        # 9th example
    {
        "role": 'user',
        "parts": [
        f"""
            You must follow these Orange arrow rules:
            'Go forward' moves forward one space in the direction the orange arrow is pointing.
            If the Orange arrow is pointing upwards and 'Turn left', the Orange arrow will point leftwards.
            If the Orange arrow is pointing upwards and 'Turn right', the Orange arrow will point rightwards.
            If the Orange arrow is pointing downwards and 'Turn left', the Orange arrow will point rightwards.
            If the Orange arrow is pointing downwards and 'Turn right', the Orange arrow will point leftwards.
            If the Orange arrow is pointing leftwards and 'Turn left', the Orange arrow will point downwards.
            If the Orange arrow is pointing leftwards and 'Turn right', the Orange arrow will point upwards.
            If the Orange arrow is pointing rightwards and 'Turn left', the Orange arrow will point upwards.
            If the Orange arrow is pointing rightwards and 'Turn right', the Orange arrow will point downwards.

            Current State:
            Based on top left (1,1),
            Orange arrow's current location : row 4 , column 4
            Orange arrow's current pointing : leftwards
            Target location's current location : row 5, column 4

            Problem : Is the orange arrow currently at the target location?
            If so, print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 so that the orange arrow is pointing to row 6 column 4.
            If not, print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach row 5 column 4.
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
            """
            The orange arrow is not currently at the target location.

            To reach the target location (row 5, column 4), the orange arrow needs to turn left.

            Action probability distribution:
            ['Turn left': 1.0, 'Turn right': 0.0, 'Go Forward': 0.0]

            [1.0, 0.0, 0.0] 
            """
        ]
    },


        # 10th example
    {
        "role": 'user',
        "parts": [
        f"""
            You must follow these Orange arrow rules:
            'Go forward' moves forward one space in the direction the orange arrow is pointing.
            If the Orange arrow is pointing upwards and 'Turn left', the Orange arrow will point leftwards.
            If the Orange arrow is pointing upwards and 'Turn right', the Orange arrow will point rightwards.
            If the Orange arrow is pointing downwards and 'Turn left', the Orange arrow will point rightwards.
            If the Orange arrow is pointing downwards and 'Turn right', the Orange arrow will point leftwards.
            If the Orange arrow is pointing leftwards and 'Turn left', the Orange arrow will point downwards.
            If the Orange arrow is pointing leftwards and 'Turn right', the Orange arrow will point upwards.
            If the Orange arrow is pointing rightwards and 'Turn left', the Orange arrow will point upwards.
            If the Orange arrow is pointing rightwards and 'Turn right', the Orange arrow will point downwards.

            Current State:
            Based on top left (1,1),
            Orange arrow's current location : row 4 , column 3
            Orange arrow's current pointing : downwards
            Target location's current location : row 5, column 3

            Problem : Is the orange arrow currently at the target location?
            If so, print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 so that the orange arrow is pointing to row 6 column 3.
            If not, print the next action's probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 for the orange arrow to reach row 5 column 3.
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
            """
            The orange arrow is not currently at the target location.

            To reach the target location (row 5, column 3), the orange arrow needs to turn left.

            Action probability distribution:
            ['Turn left': 0.0, 'Turn right': 0.0, 'Go Forward': 1.0]

            [1.0, 0.0, 0.0] 
            """
        ]
    }

]

