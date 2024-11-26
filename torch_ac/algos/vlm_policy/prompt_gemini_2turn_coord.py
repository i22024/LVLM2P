
from PIL import Image

images=[
    # 5_shot :  1 2 5 6 7
    Image.open(r"LVLM2P/algos/vlm_policy/few_shot_coordinate/few_shot1.png"),
    Image.open(r"LVLM2P/algos/vlm_policy/few_shot_coordinate/few_shot2.png"),
    Image.open(r"LVLM2P/algos/vlm_policy/few_shot_coordinate/few_shot3.png"),
    Image.open(r"LVLM2P/algos/vlm_policy/few_shot_coordinate/few_shot4.png"),
    Image.open(r"LVLM2P/algos/vlm_policy/few_shot_coordinate/few_shot5.png"),
    Image.open(r"LVLM2P/algos/vlm_policy/few_shot_coordinate/few_shot6.png"),
]
fewshot_prompt_coord=[
    # 1st example
    {
        "role": 'user',
        "parts": [images[0], 
        """
        In this 6x6 grid image, in which row and column is the purple door based on the top left?
        In which row and column is the orange arrow located, and in which direction is it facing?
        Tip: The numbers written vertically between the purple door and orange arrow are rows, and the numbers written horizontally are columns.
        On the last line, Reply your answer as a single list like this : [orange arrow's row, orange arrow's column, purple door's row, purple door's column, 'orange arrow's direction'].
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
        """
        Here's the breakdown:
        Purple door: Row 1, Column 4
        Orange arrow: Row 2, Column 2, facing rightwards
        Therefore, the answer is: 
        [2, 2, 1, 4, 'rightwards']
        """
        ]
    },
    # 2nd example
    {
        "role": 'user',
        "parts": [ images[1],
        """
        In this 6x6 grid image, in which row and column is the blue door based on the top left?
        In which row and column is the orange arrow located, and in which direction is it facing?
        Tip: The numbers written vertically between the blue door and orange arrow are rows, and the numbers written horizontally are columns.
        On the last line, Reply your answer as a single list like this : [orange arrow's row, orange arrow's column, blue door's row, blue door's column, 'orange arrow's direction'].
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
        """
        Here's the breakdown:
        Blue door: Row 4, Column 1
        Orange arrow: Row 4, Column 3, facing upwards
        Therefore, the answer is: 
        [4, 3, 4, 1, 'upwards']
        """
        ]
    },
    # 3rd example
    {
        "role": 'user',
        "parts": [images[2],
        """
        In this 6x6 grid image, in which row and column is the red door based on the top left?
        In which row and column is the orange arrow located, and in which direction is it facing?
        Tip: The numbers written vertically between the red door and orange arrow are rows, and the numbers written horizontally are columns.
        On the last line, Reply your answer as a single list like this : [orange arrow's row, orange arrow's column, red door's row, red door's column, 'orange arrow's direction'].
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
        """
        Here's the breakdown:
        Red door: Row 4, Column 6
        Orange arrow: Row 4, Column 5, facing upwards
        Therefore, the answer is: 
        [4, 5, 4, 6, 'upwards']
        """
        ]
    },

    # 4th example
    {
        "role": 'user',
        "parts": [ images[3],
        """
        In this 6x6 grid image, in which row and column is the yellow door based on the top left?
        In which row and column is the orange arrow located, and in which direction is it facing?
        Tip: The numbers written vertically between the yellow door and orange arrow are rows, and the numbers written horizontally are columns.
        On the last line, Reply your answer as a single list like this : [orange arrow's row, orange arrow's column, yellow door's row, yellow door's column, 'orange arrow's direction'].
        """
        ]
    },
    {
        "role": 'model',
        "parts": [  
        """
        Here's the breakdown:
        yellow door: Row 4, Column 1
        Orange arrow: Row 2, Column 5, facing leftwards
        Therefore, the answer is: 
        [2, 5, 4, 1, 'leftwards']
        """
        ]
    },
        # 5th example
    {
        "role": 'user',
        "parts": [ images[4],
        f"""
        In this 6x6 grid image, in which row and column is the orange door based on the top left?
        In which row and column is the orange arrow located, and in which direction is it facing?
        Tip: The numbers written vertically between the orange door and orange arrow are rows, and the numbers written horizontally are columns.
        On the last line, Reply your answer as a single list like this : [orange arrow's row, orange arrow's column, orange door's row, orange door's column, 'orange arrow's direction'].
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
        """
        Here's the breakdown:
        Orange door: Row 6, Column 3
        Orange arrow: Row 4, Column 4, facing downwards
        Therefore, the answer is: 
        [4, 4, 6, 3, 'downwards']    
        """
        ]
    },
    
        # 6th example
    {
        "role": 'user',
        "parts": [ images[5],
        f"""
        In this 6x6 grid image, in which row and column is the green door based on the top left?
        In which row and column is the orange arrow located, and in which direction is it facing?
        Tip: The numbers written vertically between the green door and orange arrow are rows, and the numbers written horizontally are columns.
        On the last line, Reply your answer as a single list like this : [orange arrow's row, orange arrow's column, green door's row, green door's column, 'orange arrow's direction'].
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
        """
        Here's the breakdown:
        Green door: Row 6, Column 4
        Orange arrow: Row 5, Column 4, facing downwards
        Therefore, the answer is: 
        [5, 4, 6, 4, 'downwards'] 
        """
        ]
    }
    ]

