from PIL import Image
import os

path_dir = "phoenix-mouthing-ECCV/fullFrames-210px-260px/RWTH-PHOENIX-Weather-Mouthing-ECCV-test/"


def generate_variations(im):
    variations = {
        'rotate': [-5, 0, 5,],
        'crop': [(i,j) for i in range(3) for j in range(3)],
    }
    for r in variations['rotate']:
        for dx,dy in variations['crop']:
            if -(5 - abs(r)) <= 4+dx-5 <= 5 - abs(r) and -(5 - abs(r)) <= 4+dy-5 <= 5 - abs(r):
                yield im.rotate(r).crop((4+dx,4+dy,221+4+dx,221+4+dy)), 1 if (r == 0 and dx == 1 and dy == 1) else 0


for d in os.listdir(path_dir):
    for im_name in os.listdir(os.path.join(path_dir,d)):
        im_path = os.path.join(path_dir,d, im_name)
        im = Image.open(im_path)
        im = im.crop((55,15,130,120))
        im = im.resize((234, 234))

        index = 0
        for (im_var, is_default) in generate_variations(im):
            if is_default:
                directory = os.path.join(path_dir+'cropped',d+'_gold')
                if not os.path.exists(directory):
                    os.makedirs(directory)
                im_var.save(os.path.join(directory, im_name), format='png')
            else:
                index += 1
                directory = os.path.join(path_dir+'cropped',d+'_{0:04d}'.format(index))
                if not os.path.exists(directory):
                    os.makedirs(directory)
                im_var.save(os.path.join(directory, im_name), format='png')

sentences = [
    line.split(' ')
    for line in open("phoenix-mouthing-ECCV/annotations/mouthing.annotations")
]

sentences2 = ''.join([
    '/'.join(path.split('/')[:-2]+['cropped']+[path.split('/')[-2]+'_{0:04d}'.format(variation_index),path.split('/')[-1]])+ ' ' + labels
    for variation_index in range(1, 11, )
    for path, labels in sentences
])
open("phoenix-mouthing-ECCV/annotations/mouthing.annotations2", 'w').write(sentences2)

sentences2gold = ''.join([
    '/'.join(path.split('/')[:-2]+['cropped']+[path.split('/')[-2]+'_gold',path.split('/')[-1]])+ ' ' + labels
    for path, labels in sentences
])
open("phoenix-mouthing-ECCV/annotations/mouthing.annotations2gold", 'w').write(sentences2gold)
