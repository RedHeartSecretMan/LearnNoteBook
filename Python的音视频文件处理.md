

# ***Python的音视频文件处理***

[toc]





## 1. ffmpeg-python

**`ffmpeg-python` 是 `ffmpeg` 的一个包装，通过 `python` 调用`ffmpeg` 的 *API*，实现高效的音视频文件处理**

<img src="././assets/Python%E7%9A%84%E9%9F%B3%E8%A7%86%E9%A2%91%E6%96%87%E4%BB%B6%E5%A4%84%E7%90%86/formula.png" alt="ffmpeg-python logo" width="60%" />



### 开始之前

**安装 `ffmpeg`**

```bash
# Linux
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```



**安装 `ffmpeg-python`**

```bash
# 方式一
pip install ffmpeg-python

# 方式二
git clone git@github.com:kkroening/ffmpeg-python.git
pip install -e ./ffmpeg-python
```



### 技术背景

**音视频文件处理流程**

**$输入文件 \stackrel{解封装}{\Longrightarrow} 已编码的数据包 \stackrel{解码}{\Longrightarrow} 被编码的帧(可进行信号处理操作) \stackrel{编码}{\Longrightarrow} 已编码的数据包 \stackrel{封装}{\Longrightarrow} 输出文件$**

<img src="././assets/Python%E7%9A%84%E9%9F%B3%E8%A7%86%E9%A2%91%E6%96%87%E4%BB%B6%E5%A4%84%E7%90%86/image-20230214175210636.png" alt="image-20230214175210636" style="zoom:50%;" />

**在 `ffmpeg-python` 中通过 `ffmpeg.input` 对音视频文件解封装，通过 `ffmpeg.filter` 对被解码的帧进行信号处理操作，通过 `ffmpeg.output` 将数据包封装成音视频文件，整个过程会构建成一个结点图，最后通过调用 `ffmpeg.run` 执行节点图上的操作**

```python
import ffmpeg

"""
输入函数
ffmpeg.input(filename, **kwargs)

其中，参数
filename - 输入文件url
kwargs - 关键字参数逐字传递给ffmpeg

目的，创建读取文件结点

另外，如果存在音频那么注册为audio属性方法，可通过.video或["v"]获取，如果存在视频那么注册为video属性方法，可通过.audio或["a"]获取
"""
kwargs = {"t":20, "f":"mp4", "acodec":"pcm"}
ffmpeg.input('in.mkv', **kwargs)
input('in.mkv', **kwargs)
stream_video = ffmpeg.input('in.mkv', **kwargs).video
stream_video = ffmpeg.input('in.mkv', **kwargs)["v"]
stream_audio = ffmpeg.input('in.mkv', **kwargs).audio
stream_audio = ffmpeg.input('in.mkv', **kwargs)["a"]


"""
过滤器函数 
ffmpeg.filter(stream_spec, filter_name, *args, **kwargs)

其中，参数
stream_spec – 一个Stream, Streams列表，或者字典映射给Stream的标签
filter_name – ffmpeg中filter的名字, 例如colorchannelmixer、crop、hflip、volume、concat等过滤器
*args – 要逐字传递给ffmpeg的参数列表
**kwargs – 要逐字传递给ffmpeg的关键字参数列表

目的，创建处理数据结点

另外，作为函数时最好用filter_避免与python的filter内建函数混淆，其中filter_会直接return在ffmpeg中的filter，带有多个输出时使用ffmpeg.filter_multi_output(stream_spec, filter_name, *args, **kwargs)
"""
ffmpeg.input('in.mp4').filter('hflip')
ffmpeg.filter(input('in.mp4'), "hflip") 


"""
输出函数 
ffmpeg.output(*streams_and_filename, **kwargs)

其中，参数
video_bitrate – 即ffmpeg中的视频码率参数 -b:v, e.g. video_bitrate=1000.
audio_bitrate – 即ffmpeg中的音频码率参数 -b:a, e.g. audio_bitrate=200.
format – 即ffmpeg中的格式参数 -f, e.g. format='mp4' (equivalent to f='mp4').

目的，创建写入文件结点
"""
kwargs = {"-b:v":1000, "-b:a":200, "f":"mp4", "acodec":"pcm", "vcodec":"rawvideo"}
streams = ffmpeg.input("in.mkv").filter('hflip')
ffmpeg.output(streams, "out.mp4", **kwargs)
streams.output(streams, "out.mp4", **kwargs)


"""
运行函数
ffmpeg.run(stream_spec, cmd='ffmpeg', capture_stdout=False, capture_stderr=False, input=None, quiet=False, overwrite_output=False)

其中，参数
capture_stdout – 是否捕捉标stdout
capture_stderr – 是否捕捉到stderr
quiet – 设置capture_stdout和capture_stderr的简写
input – 要发送到stdin的文本
overwrite_output - 是否覆盖已有文件
其中，函数返回包含捕获的stdout和stderr数据

目的，调用ffmpeg执行构建好的结点图

另外，ffmpeg.run_async(stream_spec, cmd='ffmpeg', pipe_stdin=False, pipe_stdout=False, pipe_stderr=False, quiet=False, overwrite_output=False)，异步调用ffmpeg执行构建好的节点图
"""
streams = ffmpeg.input('in.mp4').filter('hflip').output('out.mp4').run(overwrite_output=True)
ffmpeg.run(streams, overwrite_output=True)
```

**另外，音视频文件解封装后得到的数据包称为 *stream*（如视频、音频与字幕等）可以通过 `ffmpeg.probe` 查看，结点图可通过 `ffmpeg.view` 查看**

- **执行函数**

  ```python
  """
  音视频文件摘要函数 ffmpeg.probe(filename, cmd='ffprobe', **kwargs)
  
  如果ffprobe返回非零退出码，则返回一个带有通用错误消息的Error，可以通过访问异常的stderr属性来检索stderr输出
  """
  
  ffmpeg.probe("in.mkv")
  ```

- **打印输出**

  ```josn
  {'streams': [{'index': 0,
     'codec_name': 'hevc',
     'codec_long_name': 'H.265 / HEVC (High Efficiency Video Coding)',
     'profile': 'Main 10',
     'codec_type': 'video',
     'codec_time_base': '1001/24000',
     'codec_tag_string': '[0][0][0][0]',
     'codec_tag': '0x0000',
     'width': 1920,
     'height': 1080,
     'coded_width': 1920,
     'coded_height': 1080,
     'has_b_frames': 2,
     'sample_aspect_ratio': '1:1',
     'display_aspect_ratio': '16:9',
     'pix_fmt': 'yuv420p10le',
     'level': 120,
     'color_range': 'tv',
     'color_space': 'bt2020nc',
     'color_transfer': 'smpte2084',
     'color_primaries': 'bt2020',
     'refs': 1,
     'r_frame_rate': '24000/1001',
     'avg_frame_rate': '24000/1001',
     'time_base': '1/1000',
     'start_pts': 0,
     'start_time': '0.000000',
     'disposition': {'default': 1,
      'dub': 0,
      'original': 0,
      'comment': 0,
      'lyrics': 0,
      'karaoke': 0,
      'forced': 0,
      'hearing_impaired': 0,
      'visual_impaired': 0,
      'clean_effects': 0,
      'attached_pic': 0,
      'timed_thumbnails': 0},
     'tags': {'language': 'jpn',
      'BPS': '3285534',
      'DURATION': '00:24:10.032000000',
      'NUMBER_OF_FRAMES': '34766',
      'NUMBER_OF_BYTES': '595516342',
      '_STATISTICS_WRITING_APP': "mkvmerge v70.0.0 ('Caught A Lite Sneeze') 64-bit",
      '_STATISTICS_TAGS': 'BPS DURATION NUMBER_OF_FRAMES NUMBER_OF_BYTES'}},
    {'index': 1,
     'codec_name': 'eac3',
     'codec_long_name': 'ATSC A/52B (AC-3, E-AC-3)',
     'codec_type': 'audio',
     'codec_time_base': '1/48000',
     ...
     }
    },
    {'index': 2,
     'codec_name': 'eac3',
     'codec_long_name': 'ATSC A/52B (AC-3, E-AC-3)',
     'codec_type': 'audio',
     'codec_time_base': '1/48000',
     ...
     }
    },
    {'index': 3,
     'codec_name': 'subrip',
     'codec_long_name': 'SubRip subtitle',
     'codec_type': 'subtitle',
     'codec_time_base': '0/1',
     ...
     }
    },
   },
    ......
  }
  ```

- **执行函数**

  ```python
  """
  结点图可视化函数 view(detail=False, filename=None, pipe=False, **kwargs)
  
  其中，参数
  detail - 是否打印结点和边的关键信息，如stream的映射关系等
  """
  streams = ffmpeg.input('in.mp4')
  video = streams.video.filter('hflip')
  audio = streams.audio
  ffmpeg.concat(video, audio, v=1, a=1).overlay(ffmpeg.input("overlay.png")).output("out.mp4").view(detail=True)
  ```

- **打印输出**

  ![image-20230214225318620](././assets/Python%E7%9A%84%E9%9F%B3%E8%A7%86%E9%A2%91%E6%96%87%E4%BB%B6%E5%A4%84%E7%90%86/image-20230214225318620.png)





### 快速开始

#### 简单的例子，水平翻转视频

![image-20230214220940832](././assets/Python%E7%9A%84%E9%9F%B3%E8%A7%86%E9%A2%91%E6%96%87%E4%BB%B6%E5%A4%84%E7%90%86/image-20230214220940832.png)

```python
import ffmpeg

# 方式一、一般写法
stream = ffmpeg.input('input.mp4')
stream = ffmpeg.hflip(stream)
stream = ffmpeg.output(stream, 'output.mp4')
ffmpeg.run(stream)

# 方式二、数据流写法（更易读）
(
    ffmpeg
    .input('input.mp4')
    .hflip()
    .output('output.mp4')
    .run()
)
```



#### 复杂的例子，执行多滤波器

**`ffmpeg` 非常强大，但使用多个过滤器（filter）处理的信号时，命令代码显得非常粗糙。例如，将输入视频两次修剪后合并，再将水平翻转的图片覆盖在视频上，接着在上面画一个框输出视频**

![Signal graph](././assets/Python%E7%9A%84%E9%9F%B3%E8%A7%86%E9%A2%91%E6%96%87%E4%BB%B6%E5%A4%84%E7%90%86/graph1.png)

> ***图中绿色代表输入文件，黄色代表过滤器，蓝色代表输出文件，箭头代表数据流***

<img src="././assets/Python%E7%9A%84%E9%9F%B3%E8%A7%86%E9%A2%91%E6%96%87%E4%BB%B6%E5%A4%84%E7%90%86/screenshot.png" alt="Screenshot" align="middle" width="60%" />

 **`ffmpeg` 的命令很难记忆与理解琐碎**

```bash
ffmpeg -i input.mp4 -i overlay.png -filter_complex "[0]trim=start_frame=10:end_frame=20[v0];\
    [0]trim=start_frame=30:end_frame=40[v1];[v0][v1]concat=n=2[v2];[1]hflip[v3];\
    [v2][v3]overlay=eof_action=repeat[v4];[v4]drawbox=50:50:120:120:red:t=5[v5]"\
    -map [v5] output.mp4
```

 **`ffmpeg-python` 的命令简单明了**

```python
import ffmpeg

in_file = ffmpeg.input('input.mp4')
overlay_file = ffmpeg.input('overlay.png')
(
    ffmpeg
    .concat(
        in_file.trim(start_frame=10, end_frame=20),
        in_file.trim(start_frame=30, end_frame=40),
    )
    .overlay(overlay_file.hflip())
    .drawbox(50, 50, 120, 120, color='red', thickness=5)
    .output('out.mp4')
    .run()
)
```



### 更多尝试

#### 生成视频缩略图

![get_video_thumbnail](././assets/Python%E7%9A%84%E9%9F%B3%E8%A7%86%E9%A2%91%E6%96%87%E4%BB%B6%E5%A4%84%E7%90%86/get_video_thumbnail.png)

```
(
    ffmpeg
    .input(in_filename, ss=time)
    .filter('scale', width, -1)
    .output(out_filename, vframes=1)
    .run()
)
```



#### 将视频转换为numpy数组

![get-video-thumbnail graph](././assets/Python%E7%9A%84%E9%9F%B3%E8%A7%86%E9%A2%91%E6%96%87%E4%BB%B6%E5%A4%84%E7%90%86/ffmpeg-numpy.png)

```
out, _ = (
    ffmpeg
    .input('in.mp4')
    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
    .run(capture_stdout=True)
)
video = (
    np
    .frombuffer(out, np.uint8)
    .reshape([-1, height, width, 3])
)
```



#### 通过管道读取单个视频帧为jpeg

![read_frame_as_jpeg](././assets/Python%E7%9A%84%E9%9F%B3%E8%A7%86%E9%A2%91%E6%96%87%E4%BB%B6%E5%A4%84%E7%90%86/read_frame_as_jpeg.png)

```
out, _ = (
    ffmpeg
    .input(in_filename)
    .filter('select', 'gte(n,{})'.format(frame_num))
    .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
    .run(capture_stdout=True)
)
```



#### 将声音转换为原始PCM音频

![transcribe](././assets/Python%E7%9A%84%E9%9F%B3%E8%A7%86%E9%A2%91%E6%96%87%E4%BB%B6%E5%A4%84%E7%90%86/transcribe.png)

```
out, _ = (ffmpeg
    .input(in_filename, **input_kwargs)
    .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar='16k')
    .overwrite_output()
    .run(capture_stdout=True)
)
```



#### 从帧序列组装视频

![glob](././assets/Python%E7%9A%84%E9%9F%B3%E8%A7%86%E9%A2%91%E6%96%87%E4%BB%B6%E5%A4%84%E7%90%86/glob.png)

```
(
    ffmpeg
    .input('/path/to/jpegs/*.jpg', pattern_type='glob', framerate=25)
    .output('movie.mp4'
    .run()
)
```

**添加额外过滤器**

![glob-filter](././assets/Python%E7%9A%84%E9%9F%B3%E8%A7%86%E9%A2%91%E6%96%87%E4%BB%B6%E5%A4%84%E7%90%86/glob-filter.png)

```
(
    ffmpeg
    .input('/path/to/jpegs/*.jpg', pattern_type='glob', framerate=25)
    .filter('deflicker', mode='pm', size=10)
    .filter('scale', size='hd1080', force_original_aspect_ratio='increase')
    .output('movie.mp4', crf=20, preset='slower', movflags='faststart', pix_fmt='yuv420p')
    .view(filename='filter_graph')
    .run()
)
```



#### 音视频管线

![av-pipeline graph](././assets/Python%E7%9A%84%E9%9F%B3%E8%A7%86%E9%A2%91%E6%96%87%E4%BB%B6%E5%A4%84%E7%90%86/av-pipeline.png)

```
in1 = ffmpeg.input('in1.mp4')
in2 = ffmpeg.input('in2.mp4')
v1 = in1.video.hflip()
a1 = in1.audio
v2 = in2.video.filter('reverse').filter('hue', s=0)
a2 = in2.audio.filter('areverse').filter('aphaser')
joined = ffmpeg.concat(v1, a1, v2, a2, v=1, a=1).node
v3 = joined[0]
a3 = joined[1].filter('volume', 0.8)
out = ffmpeg.output(v3, a3, 'out.mp4')
out.run()
```



#### 单声道到立体声带偏移和视频

![mono-to-stereo](././assets/Python%E7%9A%84%E9%9F%B3%E8%A7%86%E9%A2%91%E6%96%87%E4%BB%B6%E5%A4%84%E7%90%86/mono-to-stereo.png)

```
audio_left = (
    ffmpeg
    .input('audio-left.wav')
    .filter('atrim', start=5)
    .filter('asetpts', 'PTS-STARTPTS')
)

audio_right = (
    ffmpeg
    .input('audio-right.wav')
    .filter('atrim', start=10)
    .filter('asetpts', 'PTS-STARTPTS')
)

input_video = ffmpeg.input('input-video.mp4')

(
    ffmpeg
    .filter((audio_left, audio_right), 'join', inputs=2, channel_layout='stereo')
    .output(input_video.video, 'output-video.mp4', shortest=None, vcodec='copy')
    .overwrite_output()
    .run()
)
```



####  Jupyter Frame Viewer

![jupyter-screenshot](././assets/Python%E7%9A%84%E9%9F%B3%E8%A7%86%E9%A2%91%E6%96%87%E4%BB%B6%E5%A4%84%E7%90%86/jupyter-screenshot.png)



#### Tensorflow Streaming

![tensorflow-stream](././assets/Python%E7%9A%84%E9%9F%B3%E8%A7%86%E9%A2%91%E6%96%87%E4%BB%B6%E5%A4%84%E7%90%86/tensorflow-stream.png)

- **`ffmpeg` 解码输入视频**
- **`tensorflow` 使用 "deep dream" 处理视频**
- **`ffmpeg` 编码输出视频**

```
process1 = (
    ffmpeg
    .input(in_filename)
    .output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=8)
    .run_async(pipe_stdout=True)
)

process2 = (
    ffmpeg
    .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
    .output(out_filename, pix_fmt='yuv420p')
    .overwrite_output()
    .run_async(pipe_stdin=True)
)

while True:
    in_bytes = process1.stdout.read(width * height * 3)
    if not in_bytes:
        break
    in_frame = (
        np
        .frombuffer(in_bytes, np.uint8)
        .reshape([height, width, 3])
    )

    # See examples/tensorflow_stream.py:
    out_frame = deep_dream.process_frame(in_frame)

    process2.stdin.write(
        out_frame
        .astype(np.uint8)
        .tobytes()
    )

process2.stdin.close()
process1.wait()
process2.wait()
```

<img src="././assets/Python%E7%9A%84%E9%9F%B3%E8%A7%86%E9%A2%91%E6%96%87%E4%BB%B6%E5%A4%84%E7%90%86/image-20230214234754387.png" alt="image-20230214234754387" style="zoom:50%;" />



#### FaceTime webcam input (OS X)

```
(
    ffmpeg
    .input('FaceTime', format='avfoundation', pix_fmt='uyvy422', framerate=30)
    .output('out.mp4', pix_fmt='yuv420p', vframes=100)
    .run()
)
```



#### Stream from a local video to HTTP server

```
video_format = "flv"
server_url = "http://127.0.0.1:8080"

process = (
    ffmpeg
    .input("input.mp4")
    .output(
        server_url, 
        codec = "copy", # use same codecs of the original video
        listen=1, # enables HTTP server
        f=video_format)
    .global_args("-re") # argument to act as a live stream
    .run()
)
```

**在终端中使用 `ffplay` 接收视频**

```
$ ffplay -f flv http://localhost:8080
```



#### Stream from RTSP server to TCP socket

```
packet_size = 4096

process = (
    ffmpeg
    .input('rtsp://%s:8554/default')
    .output('-', format='h264')
    .run_async(pipe_stdout=True)
)

while process.poll() is None:
    packet = process.stdout.read(packet_size)
    try:
        tcp_socket.send(packet)
    except socket.error:
        process.stdout.close()
        process.wait()
        break
```



#### 自定义过滤器

**虽然 `ffmpeg-python` 只直接提供了部分`ffmpeg` 的过滤器 *API*，但仍可以通过 `.filter` 运算符见解引用全部的过滤器**

```python
(
    ffmpeg
    .input('dummy.mp4')
    .filter('fps', fps=25, round='up')
    .output('dummy2.mp4')
    .run()
)
```



##### 多个输入

**接收多个输入流的过滤器可以通过将输入流作为数组传递给 `ffmpeg.filter` 使用**

```python
main = ffmpeg.input('main.mp4')
logo = ffmpeg.input('logo.png')
(
    ffmpeg
    .filter([main, logo], 'overlay', 10, 10)
    .output('out.mp4')
    .run()
)
```



##### 多个输出

**产生多个输出的过滤器可以与 `.filter_multi_output` 一起使用**

```python
split = (
    ffmpeg
    .input('in.mp4')
    .filter_multi_output('split')
)
(
    ffmpeg
    .concat(split[0], split[1].reverse())
    .output('out.mp4')
    .run()
)
```
