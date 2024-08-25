# Multimedia and Embedding
## Images
```html
The simplest way to display an img:
<img
  src="images/dinosaur.jpg"
  alt="The head and torso of a dinosaur skeleton;
          it has a large head with long sharp teeth"
  width="400"
  height="341" />
```

You can also use `figure`:
```html
<figure>
  <img
    src="images/dinosaur.jpg"
    alt="The head and torso of a dinosaur skeleton;
            it has a large head with long sharp teeth"
    width="400"
    height="341" />

  <figcaption>
    A T-Rex on display in the Manchester University Museum.
  </figcaption>
</figure>

```
A figure doesn't have to be an image. It is an independent unit of content that:
* Expresses your meaning in a compact, easy-to-grasp way.
* Could go in several places in the page's linear flow.
* Provides essential information supporting the main text.

A figure could be several images, a code snippet, audio, video, equations, a table, or something else.

## Video and Audio
```html
<video controls>
  <source src="rabbit320.mp4" type="video/mp4" />
  <source src="rabbit320.webm" type="video/webm" />
  <p>
    Your browser doesn't support this video. Here is a
    <a href="rabbit320.mp4">link to the video</a> instead.
  </p>
</video>

control:Users must be able to control video and audio playback
content: like alt attribute in img

<audio controls>
  <source src="viper.mp3" type="audio/mp3" />
  <source src="viper.ogg" type="audio/ogg" />
  <p>
    Your browser doesn't support this audio file. Here is a
    <a href="viper.mp3">link to the audio</a> instead.
  </p>
</audio>
```

To embed other type of resources, see the following elements:
* [iframe](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/iframe)
* [embed](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/embed)
* [object](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/object)

></iframe>
<iframe
  id="inlineFrameExample"
  title="Inline Frame Example"
  width="500"
  height="200"
  src="https://www.openstreetmap.org/export/embed.html?bbox=-0.004017949104309083%2C51.47612752641776%2C0.00030577182769775396%2C51.478569861898606&layer=mapnik">
</iframe>