
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

import cv2
import numpy as np


SLEEP_AFTER_TRY = 1  # seconds
NUM_ROLLOUT_WORKERS=10

CUSTOMIZED_ACTION_VOCAB = {
    0: 'Turn left', 1: 'Turn right', 2: 'Move forward',
    3: 'Pick up an object', 4: 'drop', 5: 'toggle', 6: 'Done'
}
NAVIGATION_ACTIONS = {
    0: 'Turn left_90',
    1: 'Turn right_90',
    2: 'Move forward',
}
INTERACTION_ACTIONS = {
    3: 'Pick up an object',
    4: 'drop', # Unused
    5: 'toggle', # Unused
    6: 'done', # Unused
}

def create_single_image_without_wall(obs, edit_image=False, obs_size=256):
    
    filter_wall=True

    height, width = obs.shape[:2]

    top_bottom_margin = 31  
    left_right_margin = 31  
    obs = Image.fromarray(obs.astype('uint8'), 'RGB')
    draw = ImageDraw.Draw(obs)
    draw.line((width-1-(width // 6), (width // 6), width-1-(width // 6), height-(width // 6)-1), fill="black", width=1)
    draw.line(((width // 6), height-1-(width // 6), width-(width // 6)-1, height-1-(width // 6)), fill="black", width=1)

    obs = np.array(obs)
    image_with_margin = cv2.copyMakeBorder(obs, top_bottom_margin, top_bottom_margin,
                                           left_right_margin, left_right_margin,
                                           cv2.BORDER_CONSTANT, value=[255, 255, 255])

    cell_width = (width // 6)
    cell_height = (height // 6)
    
    # 1 to 6
    for i in range(6):
        cv2.putText(image_with_margin, str(i + 1),
                    (-3+left_right_margin + cell_width * i + cell_width // 2, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.putText(image_with_margin, str(i + 1),
                    (15, 3+  top_bottom_margin + cell_height * i + cell_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.putText(image_with_margin, str(i + 1),
                    (-3+ left_right_margin + cell_width * i + cell_width // 2, height + top_bottom_margin+15 ),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.putText(image_with_margin, str(i + 1),
                    (width + left_right_margin +5, 3+ top_bottom_margin + cell_height * i + cell_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    edited_image = Image.fromarray(image_with_margin,'RGB')

    return edited_image


def create_image_from_1_sequence(seq_1, n_timesteps,timestep,instruction, edit_image=False, obs_size=256):
    obs_size=256 
    
    up_margin = 40
    midle_vertical_margin = 100
    low_margin = 40

    im_w = im_h = obs_size

    left_margin = 10
    midle_horizontal_margin = 10
    right_margin = 10

    concat_h = up_margin + obs_size + low_margin
    concat_w = left_margin + obs_size * n_timesteps + right_margin + midle_horizontal_margin * (n_timesteps - 1)
    concate_im = np.ones((concat_h, concat_w, 3), dtype=np.uint8) * 255

    # ========= Drawing boundary =========
    bp = 3  # Boundary pixels
    # SEQUENCE 1
    bot_seq1 = 10  # Bottom subtract of image for sequence 1
    concate_im[0:bp, :, :] *= 0  # Upper horizontal line
    concate_im[0: concat_h - bot_seq1, 0:bp, :] *= 0  # Left vertical line
    concate_im[concat_h - bot_seq1: concat_h - bot_seq1 + bp, :, :] *= 0  # Lower horizontal line
    concate_im[0: concat_h - bot_seq1, -bp:, :] *= 0  # Right vertical line

    # ========= Placing each state at a timestep to image =========
    for i in range(n_timesteps):
        try:
            concate_im[up_margin - 10:up_margin - 10 + im_h, left_margin * (i + 1) + im_w * i:left_margin * (i + 1) + im_w * (i + 1), :] = seq_1[i]
        except Exception as error:
            print(f"[INFO] Error when making the concat image: {error}")
            print(f"{up_margin - 10}:{up_margin - 10 + im_h}, {left_margin * (i + 1) + im_w * i}:{left_margin * (i + 1) + im_w * (i + 1)}")
            breakpoint()

    cat_im = Image.fromarray(concate_im)

    # ========= Draw label =========
    draw = ImageDraw.Draw(cat_im)
    font = ImageFont.truetype("/home/ubuntu/miniconda3/envs/old_babyai/lib/python3.7/site-packages/jupyterthemes/fonts/serif/cardoserif/Cardo-Regular.ttf", size=28, encoding="unic")


    # Draw label for timestep
    if n_timesteps > 1:
        for i in range(n_timesteps):
            if obs_size == 300:
                draw.text(((left_margin + 80) + im_h * i + left_margin * i, im_h + 35), f"Timestep {i}", (0, 0, 0), font=font)
            elif obs_size == 224:
                draw.text(((left_margin + 35) + im_h * i + left_margin * i, im_h + 35), f"Timestep {i}", (0, 0, 0), font=font)
            elif obs_size == 256:
                draw.text(((left_margin + 60) + im_h * i + left_margin * i, im_h + 35), f"Timestep {i}", (0, 0, 0), font=font)    
            else:
                raise NotImplementedError

    cat_im = np.asarray(cat_im)

    extra_top = extra_bottom = extra_left = extra_right = 15
    cat_im = np.pad(cat_im, ((extra_top, extra_bottom), (extra_left, extra_right), (0, 0)), mode='constant', constant_values=255)

    if edit_image:
        # NOTE:
        #  This is used to partially solve safety exception from Gemini "block_reason: OTHER", it might be related to the image
        # green_square
        green_square = np.zeros((cat_im.shape[0] // 2, obs_size, 3), dtype=np.uint8)
        green_square[:, :] = [100, 255, 0]
        green_square = np.concatenate((green_square, np.ones((cat_im.shape[0] // 2, obs_size, 3), dtype=np.uint8)), axis=0)
        gap_width = 50
        gap = np.ones((cat_im.shape[0], gap_width, 3), dtype=np.uint8) * 255
        cat_im_gap = np.concatenate((gap, cat_im), axis=1)
        cat_im_with_black = np.concatenate((green_square, cat_im_gap), axis=1)
        cat_im_with_black = np.concatenate((gap, cat_im_with_black), axis=1)
        cat_im = cat_im_with_black

    cat_im = Image.fromarray(cat_im)
    return cat_im




def create_image_from_2_sequences(seq_1, seq_2, n_timesteps, edit_image=False, obs_size=300):
    assert seq_1.shape == seq_2.shape, f"seq_1: {seq_1.shape}, seq_2: {seq_2.shape}"
    up_margin = 40
    midle_vertical_margin = 100
    low_margin = 40

    im_w = im_h = obs_size

    left_margin = 10
    midle_horizontal_margin = 10
    right_margin = 10

    concat_h = up_margin + obs_size + midle_vertical_margin + obs_size + low_margin
    concat_w = left_margin + obs_size * n_timesteps + right_margin + midle_horizontal_margin * (n_timesteps - 1)
    concate_im = np.ones((concat_h, concat_w, 3), dtype=np.uint8) * 255

    # ========= Drawing boundary =========
    bp = 3  # Boundary pixels
    # SEQUENCE 1
    bot_seq1 = 15  # Bottom subtract of image for sequence 1
    concate_im[0:bp, :, :] *= 0  # Upper horizontal line
    concate_im[0: int(concat_h / 2) - bot_seq1, 0:bp, :] *= 0  # Left vertical line
    concate_im[int(concat_h / 2) - bot_seq1: int(concat_h / 2) - bot_seq1 + bp, :, :] *= 0  # Lower horizontal line
    concate_im[0: int(concat_h / 2) - bot_seq1, -bp:, :] *= 0  # Right vertical line

    # SEQUENCE 2
    up_seq2 = 5  # Upper subtract of image for sequence 2
    concate_im[int(concat_h / 2) + up_seq2: int(concat_h / 2) + up_seq2 + bp, :, :] *= 0  # Upper horizontal line
    concate_im[int(concat_h / 2) + up_seq2: -1, 0:bp, :] *= 0  # Left vertical line
    concate_im[-1 - bp: -1, :, :] *= 0  # Lower horizontal line
    concate_im[int(concat_h / 2) + up_seq2: -1, - bp:, :] *= 0  # Right vertical line

    # ========= Placing each state at a timestep to image =========
    for i in range(n_timesteps):
        concate_im[up_margin:up_margin + im_h, left_margin * (i + 1) + im_w * i:left_margin * (i + 1) + im_w * (i + 1), :] = seq_1[i]
        concate_im[up_margin + im_h + midle_vertical_margin:up_margin + im_h * 2 + midle_vertical_margin,
        left_margin * (i + 1) + im_w * i:left_margin * (i + 1) + im_w * (i + 1), :] = seq_2[i]

    cat_im = Image.fromarray(concate_im)

    # ========= Draw label =========
    draw = ImageDraw.Draw(cat_im)
    font = ImageFont.truetype("FreeMono.ttf", size=28, encoding="unic")

    # Draw label of image 1 and 2
    draw.text((concate_im.shape[1] / 2 - 50, 5), "Image 1", (0, 0, 0), font=font)
    draw.text((concate_im.shape[1] / 2 - 50, im_h + midle_vertical_margin + 5), "Image 2", (0, 0, 0), font=font)

    # Draw label for timestep
    if n_timesteps > 1:
        for i in range(n_timesteps):
            if obs_size == 300:
                draw.text(((left_margin + 80) + im_h * i + left_margin * i, im_h + 40), f"Timestep {i}", (0, 0, 0), font=font)
                draw.text(((left_margin + 80) + im_h * i + left_margin * i, im_h * 2 + 143), f"Timestep {i}", (0, 0, 0), font=font)
            elif obs_size == 224:
                draw.text(((left_margin + 35) + im_h * i + left_margin * i, im_h + 40), f"Timestep {i}", (0, 0, 0), font=font)
                draw.text(((left_margin + 35) + im_h * i + left_margin * i, im_h * 2 + 143), f"Timestep {i}", (0, 0, 0), font=font)
            else:
                raise NotImplementedError

    cat_im = np.asarray(cat_im)

    extra_top = extra_bottom = extra_left = extra_right = 15
    cat_im = np.pad(cat_im, ((extra_top, extra_bottom), (extra_left, extra_right), (0, 0)), mode='constant', constant_values=255)

    if edit_image:
        # NOTE:
        #  This is used to partially solve safety exception from Gemini "block_reason: OTHER", it might be related to the image
        # green_square
        green_square = np.zeros((cat_im.shape[0] // 2, obs_size, 3), dtype=np.uint8)
        green_square[:, :] = [100, 255, 0]
        green_square = np.concatenate((green_square, np.ones((cat_im.shape[0] // 2, obs_size, 3), dtype=np.uint8)), axis=0)
        gap_width = 50
        gap = np.ones((cat_im.shape[0], gap_width, 3), dtype=np.uint8) * 255
        cat_im_gap = np.concatenate((gap, cat_im), axis=1)
        cat_im_with_black = np.concatenate((green_square, cat_im_gap), axis=1)
        cat_im_with_black = np.concatenate((gap, cat_im_with_black), axis=1)
        cat_im = cat_im_with_black

    cat_im = Image.fromarray(cat_im)
    return cat_im
    



def main():
   print("excute none")



if __name__ == '__main__':
    main()
