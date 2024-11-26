
from PIL import Image

images=[
    Image.open(r"LVLM2P/algos/vlm_policy/few_shot_examples/few_shot1.png"),
    Image.open(r"LVLM2P/algos/vlm_policy/few_shot_examples/few_shot2.png"),
    Image.open(r"LVLM2P/algos/vlm_policy/few_shot_examples/few_shot9.png"),
    Image.open(r"LVLM2P/algos/vlm_policy/few_shot_examples/few_shot6.png"),
    Image.open(r"LVLM2P/algos/vlm_policy/few_shot_examples/few_shot7.png"),
    Image.open(r"LVLM2P/algos/vlm_policy/few_shot_examples/few_shot8.png"),
    Image.open(r"LVLM2P/algos/vlm_policy/few_shot_examples/few_shot10.png"),
    Image.open(r"LVLM2P/algos/vlm_policy/few_shot_examples/few_shot11.png"),
    Image.open(r"LVLM2P/algos/vlm_policy/few_shot_examples/few_shot12.png"),
]
fewshot_prompt=[
    # 1st example
    {
        "role": 'user',
        "parts": [images[0], 
        f"""
            You must follow these Orange arrow rules:
            'Go forward' moves forward one space in the direction the orange arrow is facing.
            If the Orange arrow is facing upwards and 'Turn left', the Orange arrow will point leftwards.
            If the Orange arrow is facing upwards and 'Turn right', the Orange arrow will point rightwards.
            If the Orange arrow is facing downwards and 'Turn left', the Orange arrow will point rightwards.
            If the Orange arrow is facing downwards and 'Turn right', the Orange arrow will point leftwards.
            If the Orange arrow is facing leftwards and 'Turn left', the Orange arrow will point downwards.
            If the Orange arrow is facing leftwards and 'Turn right', the Orange arrow will point upwards.
            If the Orange arrow is facing rightwards and 'Turn left', the Orange arrow will point upwards.
            If the Orange arrow is facing rightwards and 'Turn right', the Orange arrow will point downwards.
            Remember : Except for the door, there are gray wall obstacles in the 1st row, 1st column, 6th row, and 6th column.

            Problems:
            In this 6x6 grid image, in which row and column is the purple door based on the top left?
            In which row and column is the orange arrow located, and in which direction is it facing?
            Based on the orange arrow, which direction is the purple door located, left or right, and up or down?
            If the row value of the purple door and orange arrow is the same, you must answer either left or right.
            If the column values ​of the purple door and orange arrow are the same, you must answer either upwards or downwards.
            Based on this, output the action probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 of the orange arrows to reach the purple door.
            Just focus on aligning row or column first.
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
            f"1. 'purple door' location:"
            f"The 'purple door' is located at row 1, column 4."
            f"2. Orange arrow location and direction:"
            f"The orange arrow is located at row 2, column 2, and is facing rightwards."
            f"3. 'purple door' relative to the orange arrow:"
            f"The 'purple door' is to the rightwards of the orange arrow and upwards from the orange arrow."
            f"However, we need to consider the information about the gray wall obstacles."
            f"The purple  door is in the first row, which is a gray wall except for the door itself." 
            f"This means the orange arrow cannot directly reach the purple door's row."
            f"4. To reach the 'purple door', the orange arrow needs to:"
            f"Go forward twice to reach the same column(column4) as the purple door."
            f"Turn left to face the purple door."
            f"Given these steps, we want to prioritize the initial 'Go forward' as it sets the overall direction."
            f"5. Action probability distribution:"
            f"Turn left: 0.1"
            f"Turn right: 0.0"
            f"Go Forward: 0.9"
            f"Answer:"
            f"[0.1, 0.0, 0.9]"
        ]
    },
    # 2nd example
    {
        "role": 'user',
        "parts": [ images[1],
        f"""
            You must follow these Orange arrow rules:
            'Go forward' moves forward one space in the direction the orange arrow is facing.
            If the Orange arrow is facing upwards and 'Turn left', the Orange arrow will point leftwards.
            If the Orange arrow is facing upwards and 'Turn right', the Orange arrow will point rightwards.
            If the Orange arrow is facing downwards and 'Turn left', the Orange arrow will point rightwards.
            If the Orange arrow is facing downwards and 'Turn right', the Orange arrow will point leftwards.
            If the Orange arrow is facing leftwards and 'Turn left', the Orange arrow will point downwards.
            If the Orange arrow is facing leftwards and 'Turn right', the Orange arrow will point upwards.
            If the Orange arrow is facing rightwards and 'Turn left', the Orange arrow will point upwards.
            If the Orange arrow is facing rightwards and 'Turn right', the Orange arrow will point downwards.
            Remember : Except for the door, there are 'gray wall obstacles' in the 1st row, 1st column, 6th row, and 6th column. 

            Problems:
            In this 6x6 grid image, in which row and column is the 'green door' based on the top left?
            In which row and column is the orange arrow located, and in which direction is it facing?
            Based on the orange arrow, which direction is the 'green door' located, left or right, and up or down?
            If the row value of the 'green door' and orange arrow is the same, you must answer either left or right.
            If the column values ​of the 'green door' and orange arrow are the same, you must answer either upwards or downwards.
            Based on this, output the action probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 of the orange arrows to reach the 'green door'.
            Just focus on aligning row or column first.
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
        f"1. 'green door' location:"
        f"The 'green door' is located at row 3, column 6."
        f"2. Orange arrow location and direction:"
        f"The orange arrow is located at row 4, column 3, and is facing upwards."
        f"3. 'green door' relative to the orange arrow:"
        f"The 'green door' is to the right of the orange arrow and upwards from the orange arrow."
        f"4. To reach the 'green door', the orange arrow needs to:"
        f"Go forward to align rows of Orange arrow and 'green door'."
        f"Turn right to face the 'green door'."
        f"Go forward twice to reach the 'green door'." 
        f"The most important action is 'Go Forward' because it will align the Orange arrow with the 'green door''s row."
        f"5. Action probability distribution:"
        f"Turn left: 0.1"
        f"Turn right: 0.2"
        f"Go Forward: 0.7"
        f"Answer:"
        f"[0.1, 0.2, 0.7]"
        ]
    },
    # 3rd example
    {
        "role": 'user',
        "parts": [images[2],
        f"""
            You must follow these Orange arrow rules:
            'Go forward' moves forward one space in the direction the orange arrow is facing.
            If the Orange arrow is facing upwards and 'Turn left', the Orange arrow will point leftwards.
            If the Orange arrow is facing upwards and 'Turn right', the Orange arrow will point rightwards.
            If the Orange arrow is facing downwards and 'Turn left', the Orange arrow will point rightwards.
            If the Orange arrow is facing downwards and 'Turn right', the Orange arrow will point leftwards.
            If the Orange arrow is facing leftwards and 'Turn left', the Orange arrow will point downwards.
            If the Orange arrow is facing leftwards and 'Turn right', the Orange arrow will point upwards.
            If the Orange arrow is facing rightwards and 'Turn left', the Orange arrow will point upwards.
            If the Orange arrow is facing rightwards and 'Turn right', the Orange arrow will point downwards.
            Remember : Except for the door, there are gray wall obstacles in the 1st row, 1st column, 6th row, and 6th column.

            Problems:
            In this 6x6 grid image, in which row and column is the 'blue door' based on the top left?
            In which row and column is the orange arrow located, and in which direction is it facing?
            Based on the orange arrow, which direction is the 'blue door' located, left or right, and up or down?
            If the row value of the 'blue door' and orange arrow is the same, you must answer either left or right.
            If the column values ​of the 'blue door' and orange arrow are the same, you must answer either upwards or downwards.
            Based on this, output the action probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 of the orange arrows to reach the 'blue door'.
            Just focus on aligning row or column first.
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
            f"1. 'blue door' location:"
            f"The 'blue door'' is located at row 3, column 1."
            f"2. Orange arrow location and direction:"
            f"The orange arrow is located at row 4, column 5, and is facing upwards."
            f"3. 'blue door' relative to the orange arrow:"
            f"The 'blue door' is to the leftwards of the orange arrow and upwards from the orange arrow."
            f"4. To reach the 'blue door', the orange arrow needs to:"
            f"Go Forward to align rows of Orange arrow and 'blue door'."
            f"Turn left to face the 'blue door'."
            f"Go Forward until reach the 'blue door'." 
            f"The most important action is 'Go Forward' because it will align the Orange arrow with the 'blue door''s row."
            f"5. Action probability distribution:"
            f"Turn left: 0.2"
            f"Turn right: 0.0"
            f"Go Forward: 0.8"
            f"Answer:"
            f"[0.2, 0.0, 0.8]"
        ]
    },

    # 4th example
    {
        "role": 'user',
        "parts": [ images[3],
        f"""
            You must follow these Orange arrow rules:
            'Go forward' moves forward one space in the direction the orange arrow is facing.
            If the Orange arrow is facing upwards and 'Turn left', the Orange arrow will point leftwards.
            If the Orange arrow is facing upwards and 'Turn right', the Orange arrow will point rightwards.
            If the Orange arrow is facing downwards and 'Turn left', the Orange arrow will point rightwards.
            If the Orange arrow is facing downwards and 'Turn right', the Orange arrow will point leftwards.
            If the Orange arrow is facing leftwards and 'Turn left', the Orange arrow will point downwards.
            If the Orange arrow is facing leftwards and 'Turn right', the Orange arrow will point upwards.
            If the Orange arrow is facing rightwards and 'Turn left', the Orange arrow will point upwards.
            If the Orange arrow is facing rightwards and 'Turn right', the Orange arrow will point downwards.
            Remember : Except for the door, there are gray wall obstacles in the 1st row, 1st column, 6th row, and 6th column.

            Problems:
            In this 6x6 grid image, in which row and column is the yellow door based on the top left?
            In which row and column is the orange arrow located, and in which direction is it facing?
            Based on the orange arrow, which direction is the yellow door located, left or right, and up or down?
            If the row value of the yellow door and orange arrow is the same, you must answer either left or right.
            If the column values ​of the yellow door and orange arrow are the same, you must answer either upwards or downwards.
            Based on this, output the action probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 of the orange arrows to reach the yellow door.
            Just focus on aligning row or column first.
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [  
            f"1. 'yellow door' location:"
            f"The 'yellow door' is located at row 6, column 3."
            f"2. Orange arrow location and direction:"
            f"The orange arrow is located at row 5, column 3, and is facing rightwards."
            f"3. 'yellow door' relative to the orange arrow:"
            f"The 'yellow door' is downwards from the orange arrow."
            f"4. To reach the 'yellow door', the orange arrow needs to:"
            f"Turn right to face downwards."
            f"Go forward once to reach the yellow door."
            f"Given these steps, we want to prioritize the initial 'Turn right' as it sets the overall direction."
            f"5. Action probability distribution:"
            f"Turn left: 0.1"
            f"Turn right: 0.8"
            f"Go Forward: 0.1"
            f"Answer:"
            f"[0.1, 0.8, 0.1]"
        ]
    },
        # 5th example
    {
        "role": 'user',
        "parts": [ images[4],
        f"""
            You must follow these Orange arrow rules:
            'Go forward' moves forward one space in the direction the orange arrow is facing.
            If the Orange arrow is facing upwards and 'Turn left', the Orange arrow will point leftwards.
            If the Orange arrow is facing upwards and 'Turn right', the Orange arrow will point rightwards.
            If the Orange arrow is facing downwards and 'Turn left', the Orange arrow will point rightwards.
            If the Orange arrow is facing downwards and 'Turn right', the Orange arrow will point leftwards.
            If the Orange arrow is facing leftwards and 'Turn left', the Orange arrow will point downwards.
            If the Orange arrow is facing leftwards and 'Turn right', the Orange arrow will point upwards.
            If the Orange arrow is facing rightwards and 'Turn left', the Orange arrow will point upwards.
            If the Orange arrow is facing rightwards and 'Turn right', the Orange arrow will point downwards.
            Remember : Except for the door, there are gray wall obstacles in the 1st row, 1st column, 6th row, and 6th column.

            Problems: 
            In this 6x6 grid image, in which row and column is the orange door based on the top left?
            In which row and column is the orange arrow located, and in which direction is it facing?
            Based on the orange arrow, which direction is the orange door located, left or right, and up or down?
            If the row value of the orange door and orange arrow is the same, you must answer either left or right.
            If the column values ​of the orange door and orange arrow are the same, you must answer either upwards or downwards.
            Based on this, output the action probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 of the orange arrows to reach the orange door.
            Just focus on aligning row or column first.
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
            f"1. 'orange door' location:"
            f"The 'orange door' is located at row 4, column 6."
            f"2. Orange arrow location and direction:"
            f"The orange arrow is located at row 4, column 4, and is facing upwards."
            f"3. 'orange door' relative to the orange arrow:"
            f"The 'orange door' is rightwards from the orange arrow."
            f"4. To reach the 'orange door', the orange arrow needs to:"
            f"Turn right to face rightwards."
            f"Go forward twice to reach the orange door."
            f"Given these steps, we want to prioritize the initial 'Turn right' as it sets the overall direction."
            f"5. Action probability distribution:"
            f"Turn left: 0.1"
            f"Turn right: 0.8"
            f"Go Forward: 0.1"
            f"Answer:"
            f"[0.1, 0.8, 0.1]"
        ]
    },
    
    # 6th example
    {
        "role": 'user',
        "parts": [ images[5],
        f"""
            You must follow these Orange arrow rules:
            'Go forward' moves forward one space in the direction the orange arrow is facing.
            If the Orange arrow is facing upwards and 'Turn left', the Orange arrow will point leftwards.
            If the Orange arrow is facing upwards and 'Turn right', the Orange arrow will point rightwards.
            If the Orange arrow is facing downwards and 'Turn left', the Orange arrow will point rightwards.
            If the Orange arrow is facing downwards and 'Turn right', the Orange arrow will point leftwards.
            If the Orange arrow is facing leftwards and 'Turn left', the Orange arrow will point downwards.
            If the Orange arrow is facing leftwards and 'Turn right', the Orange arrow will point upwards.
            If the Orange arrow is facing rightwards and 'Turn left', the Orange arrow will point upwards.
            If the Orange arrow is facing rightwards and 'Turn right', the Orange arrow will point downwards.
            Remember : Except for the door, there are gray wall obstacles in the 1st row, 1st column, 6th row, and 6th column.

            Problems: 
            In this 6x6 grid image, in which row and column is the orange door based on the top left?
            In which row and column is the orange arrow located, and in which direction is it facing?
            Based on the orange arrow, which direction is the orange door located, left or right, and up or down?
            If the row value of the orange door and orange arrow is the same, you must answer either left or right.
            If the column values ​of the orange door and orange arrow are the same, you must answer either upwards or downwards.
            Based on this, output the action probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 of the orange arrows to reach the orange door.
            Just focus on aligning row or column first.
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
            f"1. 'orange door' location:"
            f"The 'orange door' is located at row 4, column 1."
            f"2. Orange arrow location and direction:"
            f"The orange arrow is located at row 2, column 2, and is facing downwards."
            f"3. 'orange door' relative to the orange arrow:"
            f"The 'orange door' is leftwards from the orange arrow and downwards from the orange arrow."
            f"However, we need to consider the information about the gray wall obstacles."
            f"The orange door is in the first column, which is a gray wall except for the door itself." 
            f"This means the orange arrow cannot directly reach the orange door's column."
            f"4. To reach the 'orange door', the orange arrow needs to:"
            f"Go forward twice to reach the same column(column4) as the orange door."
            f"Turn right to face the orange door."
            f"Given these steps, we want to prioritize the initial 'Go forward' as it sets the overall direction."
            f"5. Action probability distribution:"
            f"Turn left: 0.0"
            f"Turn right: 0.1"
            f"Go Forward: 0.9" 
            f"Answer:"
            f"[0.0, 0.1, 0.9]"
        ]
    },
    # 7th example
    {
        "role": 'user',
        "parts": [ images[6],
        f"""
            You must follow these Orange arrow rules:
            'Go forward' moves forward one space in the direction the orange arrow is facing.
            If the Orange arrow is facing upwards and 'Turn left', the Orange arrow will point leftwards.
            If the Orange arrow is facing upwards and 'Turn right', the Orange arrow will point rightwards.
            If the Orange arrow is facing downwards and 'Turn left', the Orange arrow will point rightwards.
            If the Orange arrow is facing downwards and 'Turn right', the Orange arrow will point leftwards.
            If the Orange arrow is facing leftwards and 'Turn left', the Orange arrow will point downwards.
            If the Orange arrow is facing leftwards and 'Turn right', the Orange arrow will point upwards.
            If the Orange arrow is facing rightwards and 'Turn left', the Orange arrow will point upwards.
            If the Orange arrow is facing rightwards and 'Turn right', the Orange arrow will point downwards.
            Remember : Except for the door, there are gray wall obstacles in the 1st row, 1st column, 6th row, and 6th column.

            Problems: 
            In this 6x6 grid image, in which row and column is the orange door based on the top left?
            In which row and column is the orange arrow located, and in which direction is it facing?
            Based on the orange arrow, which direction is the orange door located, left or right, and up or down?
            If the row value of the orange door and orange arrow is the same, you must answer either left or right.
            If the column values ​of the orange door and orange arrow are the same, you must answer either upwards or downwards.
            Based on this, output the action probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 of the orange arrows to reach the orange door.
            Just focus on aligning row or column first.
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
            f"1. 'purple door' location:"
            f"The 'purple door' is located at row 3, column 6."
            f"2. Orange arrow location and direction:"
            f"The orange arrow is located at row 4, column 5, and is facing upwards."
            f"3. 'purple door' relative to the orange arrow:"
            f"The 'purple door' is rightwards from the orange arrow and upwards from the orange arrow."
            f"However, we need to consider the information about the gray wall obstacles."
            f"The purple door is in the column 6, which is a gray wall except for the door itself." 
            f"This means the orange arrow cannot directly reach the purple door's column."
            f"4. To reach the 'purple door', the orange arrow needs to:"
            f"Go forward to reach the same row(row3) as the purple door."
            f"Turn right to face the purple door."
            f"Given these steps, we want to prioritize the initial 'Go forward' as it sets the overall direction."
            f"5. Action probability distribution:"
            f"Turn left: 0.0"
            f"Turn right: 0.1"
            f"Go Forward: 0.9"
            f"Answer:"
            f"[0.0, 0.1, 0.9]"
        ]
    },
    # 8th example
    {
        "role": 'user',
        "parts": [ images[5],
        f"""
            You must follow these Orange arrow rules:
            'Go forward' moves forward one space in the direction the orange arrow is facing.
            If the Orange arrow is facing upwards and 'Turn left', the Orange arrow will point leftwards.
            If the Orange arrow is facing upwards and 'Turn right', the Orange arrow will point rightwards.
            If the Orange arrow is facing downwards and 'Turn left', the Orange arrow will point rightwards.
            If the Orange arrow is facing downwards and 'Turn right', the Orange arrow will point leftwards.
            If the Orange arrow is facing leftwards and 'Turn left', the Orange arrow will point downwards.
            If the Orange arrow is facing leftwards and 'Turn right', the Orange arrow will point upwards.
            If the Orange arrow is facing rightwards and 'Turn left', the Orange arrow will point upwards.
            If the Orange arrow is facing rightwards and 'Turn right', the Orange arrow will point downwards.
            Remember : Except for the door, there are gray wall obstacles in the 1st row, 1st column, 6th row, and 6th column.

            Problems: 
            In this 6x6 grid image, in which row and column is the orange door based on the top left?
            In which row and column is the orange arrow located, and in which direction is it facing?
            Based on the orange arrow, which direction is the orange door located, left or right, and up or down?
            If the row value of the orange door and orange arrow is the same, you must answer either left or right.
            If the column values ​of the orange door and orange arrow are the same, you must answer either upwards or downwards.
            Based on this, output the action probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 of the orange arrows to reach the orange door.
            Just focus on aligning row or column first.
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
            f"1. 'green door' location:"
            f"The 'green door' is located at row 1, column 4."
            f"2. Orange arrow location and direction:"
            f"The orange arrow is located at row 2, column 2, and is facing downwards."
            f"3. 'green door' relative to the orange arrow:"
            f"The 'green door' is rightwards from the orange arrow and upwards from the orange arrow."
            f"4. To reach the 'green door', the orange arrow needs to:"
            f"Turn left to reach the same column(column4) as the green door."
            f"And Go Forward 2 steps to align the green door."
            f"And Turn left to face the green door."
            f"Given these steps, we want to prioritize the initial 'Turn left' as it sets the overall direction."
            f"5. Action probability distribution:"
            f"Turn left: 0.9"
            f"Turn right: 0.1"
            f"Go Forward: 0.0"
            f"Answer:"
            f"[0.9, 0.1, 0.0]"
        ]
    }
,
    # 9th example
    {
        "role": 'user',
        "parts": [ images[7],
        f"""
            You must follow these Orange arrow rules:
            'Go forward' moves forward one space in the direction the orange arrow is facing.
            If the Orange arrow is facing upwards and 'Turn left', the Orange arrow will point leftwards.
            If the Orange arrow is facing upwards and 'Turn right', the Orange arrow will point rightwards.
            If the Orange arrow is facing downwards and 'Turn left', the Orange arrow will point rightwards.
            If the Orange arrow is facing downwards and 'Turn right', the Orange arrow will point leftwards.
            If the Orange arrow is facing leftwards and 'Turn left', the Orange arrow will point downwards.
            If the Orange arrow is facing leftwards and 'Turn right', the Orange arrow will point upwards.
            If the Orange arrow is facing rightwards and 'Turn left', the Orange arrow will point upwards.
            If the Orange arrow is facing rightwards and 'Turn right', the Orange arrow will point downwards.
            Remember : Except for the door, there are gray wall obstacles in the 1st row, 1st column, 6th row, and 6th column.

            Problems:
            In this 6x6 grid image, in which row and column is the purple door based on the top left?
            In which row and column is the orange arrow located, and in which direction is it facing?
            Based on the orange arrow, which direction is the purple door located, left or right, and up or down?
            If the row value of the purple door and orange arrow is the same, you must answer either left or right.
            If the column values ​of the purple door and orange arrow are the same, you must answer either upwards or downwards.
            Based on this, output the action probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 of the orange arrows to reach the purple door.
            Just focus on aligning row or column first.
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
            f"1. 'purple door' location:"
            f"The 'purple door' is located at row 1, column 4."
            f"2. Orange arrow location and direction:"
            f"The orange arrow is located at row 2, column 4, and is facing rightwards."
            f"3. 'purple door' relative to the orange arrow:"
            f"The 'purple door' is upwards from the orange arrow."
            f"4. To reach the 'purple door', the orange arrow needs to:"
            f"Turn left to face the purple door."
            f"Given these steps, we want to prioritize the initial 'Turn left' as it sets the overall direction."
            f"5. Action probability distribution:"
            f"Turn left: 1.0"
            f"Turn right: 0.0"
            f"Go Forward: 0.0"
            f"Answer:"
            f"[1.0, 0.0, 0.0]"
        ]
    },
    # 10th example
    {
        "role": 'user',
        "parts": [ images[8],
        f"""
            You must follow these Orange arrow rules:
            'Go forward' moves forward one space in the direction the orange arrow is facing.
            If the Orange arrow is facing upwards and 'Turn left', the Orange arrow will point leftwards.
            If the Orange arrow is facing upwards and 'Turn right', the Orange arrow will point rightwards.
            If the Orange arrow is facing downwards and 'Turn left', the Orange arrow will point rightwards.
            If the Orange arrow is facing downwards and 'Turn right', the Orange arrow will point leftwards.
            If the Orange arrow is facing leftwards and 'Turn left', the Orange arrow will point downwards.
            If the Orange arrow is facing leftwards and 'Turn right', the Orange arrow will point upwards.
            If the Orange arrow is facing rightwards and 'Turn left', the Orange arrow will point upwards.
            If the Orange arrow is facing rightwards and 'Turn right', the Orange arrow will point downwards.
            Remember : Except for the door, there are gray wall obstacles in the 1st row, 1st column, 6th row, and 6th column.
 
            Problems:
            In this 6x6 grid image, in which row and column is the red door based on the top left?
            In which row and column is the orange arrow located, and in which direction is it facing?
            Based on the orange arrow, which direction is the red door located, left or right, and up or down?
            If the row value of the red door and orange arrow is the same, you must answer either left or right.
            If the column values ​of the red door and orange arrow are the same, you must answer either upwards or downwards.
            Based on this, output the action probability distribution among ['Turn left', 'Turn right', 'Go Forward'] with sum of 1 of the orange arrows to reach the red door.
            Just focus on aligning row or column first.
            On the last line, Reply your answer as a single list.
        """
        ]
    },
    {
        "role": 'model',
        "parts": [
            f"1. 'red door' location:"
            f"The 'red door' is located at row 6, column 4."
            f"2. Orange arrow location and direction:"
            f"The orange arrow is located at row 4, column 4, and is facing upwards."
            f"3. 'red door' relative to the orange arrow:"
            f"The 'red door' is downwards from the orange arrow."
            f"4. To reach the 'red door', the orange arrow needs to:"
            f"Turn left to face the red door."
            f"Given these steps, we want to prioritize the initial 'Turn left' and 'Turn right' as it sets the overall direction."
            f"5. Action probability distribution:"
            f"Turn left: 0.5"
            f"Turn right: 0.5"
            f"Go Forward: 0.0"
            f"Answer:"
            f"[0.5, 0.5, 0.0]"
        ]
    }

]

