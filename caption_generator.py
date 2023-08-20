import captioner_utils
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import einops
import numpy as np
import captioner 
import pickle

feature_extractor = captioner_utils.load_feature_extractor()

tokenizer = captioner_utils.load_tokenizer()
tokenizer.load_assets("/assets")

output_layer = captioner_utils.TokenOutput(tokenizer,
                                           banned_tokens=('', '[UNK]', '[START]'))
with open("/assets/token-output-layer.pkl", "rb") as tok_out_file:
    tok_data = pickle.load(tok_out_file)
output_layer = tok_out_file.from_config()
caption_model = captioner.Captioner(tokenizer,
                                    feature_extractor,
                                    )
def show_run(caption_model, image_path, temperature=0.0):
    image = captioner_utils.load_image(image_path)
    result_txt = caption_model.simple_gen(image, temperature)

    plt.imshow(Image.open(image_path))
    plt.axis("off")
    plt.suptitle(result_txt)
    plt.show()

    str_tokens = result_txt.split()
    str_tokens.append('[END]')

    attention_maps = [layer.last_attention_scores for layer in caption_model.decoder_layers]
    attention_maps = tf.concat(attention_maps, axis=0)
    attention_maps = einops.reduce(
        attention_maps,
        'batch heads sequence (height width) -> sequence height width',
        height=7, width=7,
        reduction='mean')

    fig = plt.figure(figsize=(16, 9))
    len_result = len(str_tokens)
    titles = []
    for i in range(len_result):
      map = attention_maps[i]
      grid_size = max(int(np.ceil(len_result/2)), 2)
      ax = fig.add_subplot(3, grid_size, i+1)
      titles.append(ax.set_title(str_tokens[i]))
      img = ax.imshow(image/255)
      ax.imshow(map, cmap='gray', alpha=0.6, extent=img.get_extent(),
                clim=[0.0, np.max(map)])
      ax.axis("off")

    plt.tight_layout()
    t = plt.suptitle(result_txt)
    t.set_y(1.05)

    return result_txt