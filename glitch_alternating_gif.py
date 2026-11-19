"""
两张图片交替生成故障动图（GIF）

效果组成：
  * 横向条状故障：整幅图的行块随机横向偏移（glitch_amount 控制强度）
  * 闪烁方块故障：随机位置出现方块，逐帧变化（位移/切条/通道错位/反色/反差色）
  * 横向故障线：随机高度的细横线，做整行位移、反差色、通道错位或局部撕裂
  * RGB 通道偏移（color_offset）与扫描线（scan_lines）

方块和故障线的颜色取当前图片主色的互补色（反差色）。

同一张图连续停留 --hold 帧才切换；组内横向条状故障和反差色保持稳定，
方块和横向故障线则逐帧重新随机，形成闪烁感。

用法:
  python glitch_alternating_gif.py a.png b.png -o out.gif
  python glitch_alternating_gif.py a.png b.png --frames 16 --hold 4 --duration 50 --lines 3
  python glitch_alternating_gif.py a.png b.png --size 640x480 --blocks 3 --lines 4
"""

import argparse
import os
import random

import numpy as np
from PIL import Image

from glitch_this import ImageGlitcher


def parse_size(value: str):
    # 解析 "宽x高" 形式的尺寸参数
    try:
        width, height = value.lower().split('x')
        width, height = int(width), int(height)
    except ValueError:
        raise argparse.ArgumentTypeError('尺寸格式应为 宽x高，例如 640x480')
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError('宽度和高度必须是正整数')
    return width, height


def load_pair(path_a: str, path_b: str, size=None):
    # 载入两张图，并统一成同一尺寸，否则无法逐帧合成 GIF
    img_a = Image.open(path_a).convert('RGB')
    img_b = Image.open(path_b).convert('RGB')

    target_size = size or img_a.size
    if img_a.size != target_size:
        img_a = img_a.resize(target_size, Image.LANCZOS)
    if img_b.size != target_size:
        img_b = img_b.resize(target_size, Image.LANCZOS)

    return img_a, img_b


def clamp_amount(amount: float, glitcher: ImageGlitcher) -> float:
    return min(max(amount, glitcher.glitch_min), glitcher.glitch_max)


def get_contrast_color(img: Image.Image):
    """
    取图片的主色，再换算成互补色（反差色）

    主色用缩小 + 少量颜色量化来估计，避免平均色发灰
    """
    small = img.resize((64, 64))
    quantized = small.quantize(colors=8, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette()

    # getcolors() 返回 [(像素数, 调色板索引), ...]，取出现最多的颜色
    _, index = max(quantized.getcolors())
    red, green, blue = palette[index * 3:index * 3 + 3]
    return (255 - red, 255 - green, 255 - blue)


def apply_block_glitches(img: Image.Image, rng: random.Random, max_blocks: int,
                         contrast_color):
    """
    在随机位置叠加方块型故障

    每个方块随机选择一种效果：
      * roll_x / roll_y：方块整体横向/纵向位移并回卷
      * strips：方块内部再切成若干横向细条，每条独立横向偏移
      * channel：只位移其中一个颜色通道，形成色差
      * invert：方块反色
      * contrast：方块染成图片主色的互补色（反差色）

    返回处理后的图片和实际使用的方块数量
    """
    arr = np.array(img)
    height, width = arr.shape[:2]

    # 方块边长约为短边的 1/10 到 1/3，保证足够醒目
    side_min = max(24, min(width, height) // 10)
    side_max = max(side_min + 1, min(width, height) // 3)

    block_count = rng.randint(1, max_blocks)
    for _ in range(block_count):
        block_w = min(rng.randint(side_min, side_max), width)
        block_h = min(rng.randint(side_min, side_max), height)

        x = rng.randint(0, width - block_w)
        y = rng.randint(0, height - block_h)
        block = arr[y:y + block_h, x:x + block_w].copy()

        effect = rng.choice(
            ('roll_x', 'roll_y', 'strips', 'channel', 'invert', 'contrast'))
        if effect == 'roll_x':
            shift = rng.randint(1, max(1, block_w - 1))
            block = np.roll(block, shift, axis=1)
        elif effect == 'roll_y':
            shift = rng.randint(1, max(1, block_h - 1))
            block = np.roll(block, shift, axis=0)
        elif effect == 'strips':
            # 方块内再切细条，每条单独横向偏移
            band_count = rng.randint(2, 6)
            edges = sorted(rng.sample(range(1, block_h),
                                      min(band_count - 1, block_h - 1)))
            start = 0
            for end in edges + [block_h]:
                max_shift = max(1, block_w // 2)
                block[start:end] = np.roll(
                    block[start:end], rng.randint(-max_shift, max_shift), axis=1)
                start = end
        elif effect == 'channel':
            # 只偏移一个颜色通道
            channel = rng.randrange(3)
            block[..., channel] = np.roll(
                block[..., channel],
                rng.randint(1, max(1, block_w - 1)), axis=1)
        elif effect == 'invert':
            block = 255 - block
        else:
            # 反差色块：保留一点原纹理，其余压成图片主色的互补色
            block = np.clip(
                block.astype(np.float32) * 0.3
                + np.asarray(contrast_color, dtype=np.float32) * 0.7,
                0, 255).astype(np.uint8)

        arr[y:y + block_h, x:x + block_w] = block

    return Image.fromarray(arr), block_count


def apply_glitch_lines(img: Image.Image, rng: random.Random, max_lines: int,
                       contrast_color):
    """
    在随机高度叠加横向故障线，模拟信号干扰

    每条线随机选择一种效果：
      * roll：整行横向位移并回卷
      * contrast：整行染成图片主色的互补色（反差色）
      * channel：只位移一个颜色通道
      * tear：把该行的一段替换成图像中另一行的一段

    返回处理后的图片和实际使用的故障线数量
    """
    arr = np.array(img)
    height, width = arr.shape[:2]

    line_count = rng.randint(1, max_lines)
    max_thickness = max(1, height // 240 + 1)
    for _ in range(line_count):
        y = rng.randint(0, height - 1)
        thickness = rng.randint(1, max_thickness)
        y_end = min(y + thickness, height)
        line = arr[y:y_end]

        effect = rng.choice(('roll', 'contrast', 'channel', 'tear'))
        if effect == 'roll':
            shift = rng.randint(max(1, width // 20), max(2, width // 3))
            if rng.random() < 0.5:
                shift = -shift
            line[:] = np.roll(line, shift, axis=1)
        elif effect == 'contrast':
            # 反差色线：保留少量原纹理，整体染成互补色
            line[:] = np.clip(
                line.astype(np.float32) * 0.25
                + np.asarray(contrast_color, dtype=np.float32) * 0.75,
                0, 255).astype(np.uint8)
        elif effect == 'channel':
            channel = rng.randrange(3)
            line[..., channel] = np.roll(
                line[..., channel], rng.randint(2, max(3, width // 6)), axis=1)
        else:
            # 局部撕裂：用另一行的一段覆盖这一行
            src_y = rng.randint(0, height - 1)
            src = arr[src_y:min(src_y + thickness, height)].copy()
            src = np.resize(src, line.shape) if src.shape[0] < line.shape[0] else src[:line.shape[0]]
            seg_w = rng.randint(max(1, width // 8), width)
            x0 = rng.randint(0, width - seg_w)
            line[:, x0:x0 + seg_w] = src[:, x0:x0 + seg_w]

    return Image.fromarray(arr), line_count


def apply_scrolling_scanlines(img: Image.Image, offset: int, period: int = 4,
                              band: int = None) -> Image.Image:
    """
    叠加可以竖向滚动的扫描线

    offset: 当前帧的滚动偏移（行），每帧 +1 扫描线就向下移动 1 像素
    period: 扫描线周期，例如 4 表示每 4 行重复一次
    band: 每个周期里压黑的行数，默认取 period 的一半
    """
    if band is None:
        band = max(1, period // 2)
    arr = np.array(img)
    height = arr.shape[0]
    rows = (np.arange(height) - offset) % period
    arr[rows < band, :, :3] = 0
    return Image.fromarray(arr)


def quantize_frames(frames):
    """
    用所有帧拼成的缩略图生成一份共享调色板，再统一量化

    这样保持不动的区域在相邻帧里颜色完全一致，
    只有方块和故障线的位置会变化
    """
    thumb_size = (160, 120)
    columns = min(4, len(frames))
    rows = (len(frames) + columns - 1) // columns

    sheet = Image.new('RGB', (columns * thumb_size[0], rows * thumb_size[1]))
    for index, frame in enumerate(frames):
        thumb = frame.resize(thumb_size, Image.BILINEAR)
        sheet.paste(thumb, ((index % columns) * thumb_size[0],
                            (index // columns) * thumb_size[1]))

    palette = sheet.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    return [frame.quantize(palette=palette, dither=Image.Dither.NONE)
            for frame in frames]


def build_frames(images, names, frames: int, hold: int, level: float,
                 level_step: float, seed, color_offset: bool,
                 scan_lines: bool, max_blocks: int, max_lines: int,
                 scan_period: int = 4):
    glitcher = ImageGlitcher()
    glitched_frames = []

    current_group = -1
    glitch_seed = None
    contrast_color = (255, 255, 255)

    for i in range(frames):
        # 关键点：同一组内图片保持不变，切换变慢但循环不中断
        group = i // hold
        img_index = group % len(images)
        base_img = images[img_index]

        # 同一组共用故障种子，让横向条状故障保持稳定；
        # 方块和故障线则逐帧换种子，形成闪烁
        if group != current_group:
            current_group = group
            glitch_seed = None if seed is None else seed + group
            # 取当前这张图的反差色，整组保持一致
            contrast_color = get_contrast_color(base_img)
        frame_rng = random.Random(None if seed is None else seed * 1000 + i)

        # 让故障强度按组摆动，避免每次切换看起来完全一样
        amount = clamp_amount(level + level_step * ((group % 3) - 1), glitcher)

        glitch_img = glitcher.glitch_image(
            base_img,
            amount,
            seed=glitch_seed,
            color_offset=color_offset,
            # 库自带的扫描线是固定的，这里关掉，改用下面可滚动的版本
            scan_lines=False,
        )

        if scan_lines:
            # 每帧下移 1 像素，走满一个周期后回到起点
            glitch_img = apply_scrolling_scanlines(
                glitch_img, i % scan_period, scan_period)

        block_count = 0
        if max_blocks > 0:
            glitch_img, block_count = apply_block_glitches(
                glitch_img, frame_rng, max_blocks, contrast_color)

        line_count = 0
        if max_lines > 0:
            glitch_img, line_count = apply_glitch_lines(
                glitch_img, frame_rng, max_lines, contrast_color)

        # 先保留 RGB，等所有帧生成完再用统一调色板量化，
        # 否则每帧各自的调色板会让静止区域整体轻微变色，看起来像全屏闪烁
        glitched_frames.append(glitch_img)
        print(f'  第 {i + 1}/{frames} 帧完成 (glitch_amount={amount:.1f}, '
              f'来源={os.path.basename(names[img_index])}, '
              f'方块={block_count}, 故障线={line_count}, '
              f'反差色={contrast_color})', flush=True)

    return glitched_frames


def main():
    parser = argparse.ArgumentParser(description='两张图片交替生成故障动图')
    parser.add_argument('img_a', help='第一张图片路径')
    parser.add_argument('img_b', help='第二张图片路径')
    parser.add_argument('-o', '--output', default='glitch_alternating.gif',
                        help='输出 GIF 路径，默认 glitch_alternating.gif')
    parser.add_argument('--frames', type=int, default=16,
                        help='总帧数，默认 16（取 图片数*hold 的整数倍可无缝循环）')
    parser.add_argument('--hold', type=int, default=4,
                        help='每张图连续停留的帧数，越大交替越慢，默认 4')
    parser.add_argument('--duration', type=int, default=50,
                        help='每帧显示时长（毫秒），默认 50')
    parser.add_argument('--level', type=float, default=4.0,
                        help='基础故障强度，范围 0.1-10，默认 4')
    parser.add_argument('--level-step', type=float, default=1.0,
                        help='故障强度逐组摆动幅度，默认 1')
    parser.add_argument('--blocks', type=int, default=3,
                        help='每帧最多出现几个闪烁方块，0 表示关闭，默认 3')
    parser.add_argument('--lines', type=int, default=3,
                        help='每帧最多出现几条横向故障线，0 表示关闭，默认 3')
    parser.add_argument('--scan-period', type=int, default=4,
                        help='扫描线周期（每多少行循环一次），默认 4')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子，固定后每次生成结果一致，默认 42')
    parser.add_argument('--size', type=parse_size, default=None,
                        help='统一尺寸，格式 宽x高，默认沿用第一张图尺寸')
    parser.add_argument('--no-color-offset', action='store_true',
                        help='关闭 RGB 通道偏移')
    parser.add_argument('--no-scan-lines', action='store_true',
                        help='关闭扫描线')
    args = parser.parse_args()

    if args.frames <= 0:
        parser.error('--frames 必须是正整数')
    if args.hold <= 0:
        parser.error('--hold 必须是正整数')
    if args.hold > args.frames:
        parser.error('--hold 不能大于 --frames')
    if args.blocks < 0:
        parser.error('--blocks 不能是负数')
    if args.lines < 0:
        parser.error('--lines 不能是负数')
    if args.scan_period < 2:
        parser.error('--scan-period 必须是不小于 2 的整数')
    if not 0.1 <= args.level <= 10:
        parser.error('--level 必须在 0.1 到 10 之间')

    cycle = args.hold * 2
    if args.frames % cycle != 0:
        print(f'提示: 帧数 {args.frames} 不是 {cycle}（hold*2）的整数倍，'
              f'循环接缝处会提前切换图片')

    if not args.no_scan_lines and args.frames % args.scan_period != 0:
        print(f'提示: 帧数 {args.frames} 不是扫描线周期 {args.scan_period} 的整数倍，'
              f'循环接缝处扫描线会跳一下')

    print('载入图片...')
    images = load_pair(args.img_a, args.img_b, args.size)
    print(f'  输出尺寸: {images[0].size[0]}x{images[0].size[1]}')

    print('生成故障帧...')
    frames = build_frames(
        images,
        [args.img_a, args.img_b],
        args.frames,
        args.hold,
        args.level,
        args.level_step,
        args.seed,
        color_offset=not args.no_color_offset,
        scan_lines=not args.no_scan_lines,
        max_blocks=args.blocks,
        max_lines=args.lines,
        scan_period=args.scan_period,
    )

    print(f'合成 GIF -> {args.output}')
    frames = quantize_frames(frames)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration,
        loop=0,
        disposal=2,
        optimize=True,
    )
    print(f'完成，共 {len(frames)} 帧')


if __name__ == '__main__':
    main()
