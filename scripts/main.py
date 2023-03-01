import functools
import os.path
import time


#import modules.img2img
import modules.txt2img
import modules.scripts as scripts

from modules import shared, script_callbacks
import gradio as gr


import os.path

import cv2


def video_to_images(frames, video_path, out_path):
    cap = cv2.VideoCapture(video_path)
    judge = cap.isOpened()
    if not judge:
        raise ValueError("Can't open video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if frames > fps:
        frames = fps

    skip = fps // frames
    count = 1
    fs = 1

    while (judge):
        flag, frame = cap.read()
        if not flag:
            break
        else:
            if fs % skip == 0:
                imgname = 'jpgs_' + str(count).rjust(3, '0') + ".jpg"
                newPath = os.path.join(out_path, imgname)
                cv2.imwrite(newPath, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                count += 1
        fs += 1
    cap.release()
    return frames


def images_to_video(frames, w, h, in_path, out_path):
    images = os.listdir(in_path)
    images = [file for file in images if file.endswith('.jpg')]
    if len(images) == 0:
        raise FileNotFoundError('not images')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(out_path, fourcc, frames, (w, h))

    images.sort(key=lambda x: int(x.replace('.jpg', '').replace('jpgs_', '')))

    for image in images:
        p = os.path.join(in_path, image)
        img = cv2.imread(p)
        video.write(img)
        del img
    video.release()
    return out_path


def hook(hookfunc, oldfunc):
    def foo(*args, **kwargs):
        return hookfunc(*args, **kwargs)

    return foo


class Script(scripts.Script):
    def __init__(self) -> None:
        self.debug = False
        self.video_file = ''
        self.movie_frames = 0
        self.enabled = False
        self.video_file_component = None
        #modules.img2img.img2img = hook(self.mov2movByPose, modules.img2img.img2img)
        modules.txt2img.txt2img = hook(self.mov2movByPose, modules.txt2img.txt2img)
        super().__init__()

    def title(self):
        #return "mov2movByPose"
        return "mov2movByPose"

    def show(self, is_txt2img):
        return scripts.AlwaysVisible

    def noise_multiplier_change(self, noise_multiplier):
        shared.opts.data['initial_noise_multiplier'] = noise_multiplier

    def color_correction_change(self, color_correction):
        shared.opts.data['img2img_color_correction'] = color_correction

    #def ui(self, is_img2img):
    def ui(self, is_txt2img):
        #if is_img2img:
        if is_txt2img:
            with gr.Group():
                with gr.Accordion("mov2movByPose", open=False):
                    self.video_file_component = gr.Video()

                    enabled = gr.Checkbox(value=False, label="Enabled")
                    with gr.Row():
                        noise_multiplier = gr.Slider(minimum=0,
                                                     maximum=1.5,
                                                     step=0.1,
                                                     label='System: Noise multiplier for img2img',
                                                     elem_id='mm_img2img_noise_multiplier',
                                                     value=0)

                        color_correction = gr.Checkbox(
                            value=False,
                            elem_id='mm_img2img_color_correction',
                            label='System: Apply color correction to img2img results to match original colors.')

                    with gr.Row():
                        movie_frames = gr.Slider(minimum=10,
                                                 maximum=60,
                                                 step=1,
                                                 label='Movie Frames',
                                                 elem_id='mm_img2img_movie_frames',
                                                 value=30)

                        button_apply = gr.Button(value='Apply', variant='secondary', elem_id='mm_img2img_apply')

            #noise_multiplier.change(fn=self.noise_multiplier_change, inputs=[noise_multiplier])
            #color_correction.change(fn=self.color_correction_change, inputs=[color_correction])

            button_apply.click(fn=self.do_apply,
                               inputs=[enabled, self.video_file_component, movie_frames, noise_multiplier,
                                       color_correction])

            return [enabled, self.video_file_component, noise_multiplier, color_correction, movie_frames, button_apply]

    def do_apply(self, enabled, video_file, movie_frames, noise_multiplier, color_correction):
        self.enabled = enabled
        self.video_file = video_file
        self.movie_frames = movie_frames
        #shared.opts.data['initial_noise_multiplier'] = noise_multiplier
        #shared.opts.data['img2img_color_correction'] = color_correction

    # def mov2movByPose(self, id_task: str, mode: int, prompt: str, negative_prompt: str, prompt_styles, init_img,
    #             sketch,
    #             init_img_with_mask, inpaint_color_sketch, inpaint_color_sketch_orig, init_img_inpaint,
    #             init_mask_inpaint,
    #             steps: int, sampler_index: int, mask_blur: int, mask_alpha: float, inpainting_fill: int,
    #             restore_faces: bool, tiling: bool, n_iter: int, batch_size: int, cfg_scale: float,
    #             image_cfg_scale: float,
    #             denoising_strength: float, seed: int, subseed: int, subseed_strength: float, seed_resize_from_h: int,
    #             seed_resize_from_w: int, seed_enable_extras: bool, height: int, width: int, resize_mode: int,
    #             inpaint_full_res: bool, inpaint_full_res_padding: int, inpainting_mask_invert: int,
    #             img2img_batch_input_dir: str, img2img_batch_output_dir: str, img2img_batch_inpaint_mask_dir: str,
    #             override_settings_texts, *args):
    def mov2movByPose(self, id_task: str, prompt: str, negative_prompt: str,
                prompt_styles, steps: int, sampler_index: int, restore_faces: bool,
                tiling: bool, n_iter: int, batch_size: int, cfg_scale: float, seed: int,
                subseed: int, subseed_strength: float, seed_resize_from_h: int,
                seed_resize_from_w: int, seed_enable_extras: bool, height: int,
                width: int, enable_hr: bool, denoising_strength: float, hr_scale: float,
                hr_upscaler: str, hr_second_pass_steps: int, hr_resize_x: int,
                hr_resize_y: int, override_settings_texts, *args):
        if self.enabled:
            if not self.video_file:
                raise ValueError('not video file')

            # 路径处理
            if self.debug:
                print('开始处理')

            mov2movByPose_images_path = os.path.join(scripts.basedir(), 'outputs', 'mov2movByPose-images')
            if self.debug:
                print(f'mov2movByPose_images_path：{mov2movByPose_images_path}')

            if not os.path.exists(mov2movByPose_images_path):
                os.mkdir(mov2movByPose_images_path)

            current_mov2movByPose_images_path = os.path.join(mov2movByPose_images_path, str(int(time.time())))
            os.mkdir(current_mov2movByPose_images_path)

            if self.debug:
                print(f'current_mov2movByPose_images_path：{current_mov2movByPose_images_path}')

            v2i_path = os.path.join(current_mov2movByPose_images_path, 'In')
            i2v_path = os.path.join(current_mov2movByPose_images_path, 'Out')
            os.mkdir(v2i_path)
            os.mkdir(i2v_path)

            # 首先把视频转换成图片
            frames = video_to_images(self.movie_frames, self.video_file, v2i_path)

            if self.debug:
                print(f'帧数：{frames},开始batch处理')
            mode = 5
            img2img_batch_input_dir = v2i_path
            img2img_batch_output_dir = i2v_path

        # result = img2img(id_task, mode, prompt, negative_prompt, prompt_styles, init_img, sketch,
        #                  init_img_with_mask, inpaint_color_sketch, inpaint_color_sketch_orig, init_img_inpaint,
        #                  init_mask_inpaint,
        #                  steps, sampler_index, mask_blur, mask_alpha, inpainting_fill,
        #                  restore_faces, tiling, n_iter, batch_size, cfg_scale,
        #                  image_cfg_scale,
        #                  denoising_strength, seed, subseed, subseed_strength, seed_resize_from_h,
        #                  seed_resize_from_w, seed_enable_extras, height, width, resize_mode,
        #                  inpaint_full_res, inpaint_full_res_padding, inpainting_mask_invert,
        #                  img2img_batch_input_dir, img2img_batch_output_dir, img2img_batch_inpaint_mask_dir,
        #                  override_settings_texts, *args)


        # def txt2img(id_task: str, prompt: str, negative_prompt: str, prompt_styles, steps: int, sampler_index: int,
        #             restore_faces: bool, tiling: bool, n_iter: int, batch_size: int, cfg_scale: float, seed: int,
        #             subseed: int, subseed_strength: float, seed_resize_from_h: int, seed_resize_from_w: int,
        #             seed_enable_extras: bool, height: int, width: int, enable_hr: bool, denoising_strength: float,
        #             hr_scale: float, hr_upscaler: str, hr_second_pass_steps: int, hr_resize_x: int, hr_resize_y: int,
        #             override_settings_texts, *args):
        result = txt2img(id_task, prompt, negative_prompt, prompt_styles, steps, sampler_index,
                    restore_faces, tiling, n_iter, batch_size, cfg_scale, seed,
                    subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w,
                    seed_enable_extras, height, width, enable_hr, denoising_strength,
                    hr_scale, hr_upscaler, hr_second_pass_steps, hr_resize_x, hr_resize_y,
                    override_settings_texts, *args)

        if self.enabled:
            print('generating mp4 file')
            movie = images_to_video(self.movie_frames, width, height, i2v_path, os.path.join(i2v_path, 'movie.mp4'))
            print(f'generate mp4 file:{movie}')
        return result


txt2img = modules.txt2img.txt2img


def on_script_unloaded():
    modules.txt2img.txt2img = txt2img


script_callbacks.on_script_unloaded(on_script_unloaded)
