# KSS: Kitchen Silence Splitter ![](kss_github_logo_v2.png)

The Kitchen Silence Splitter, or KSS, is a python program that trims videos based on their volume.

## Installation - coming soon!

Use [pip](https://pip.pypa.io/en/stable/) to install packages.

```bash
pip install [package_name]
```

## Usage

### Preparing the Workspace

Create a directory in the same folder as __kss2.py__ named __footage__, and add your video files to a folder named input.

### Setting Up the Environment

Ensure you have __ffmpeg-python__ and __moviepy__, as well as:
 - opencv-python
 - pydub

### Running the Program

Find the most recent version (__kss3.py__) and run it.  Don't forget to check options.txt!

```python
python kss3.py
```

The completed file will show up in the folder named __output__ by default.

## Contributing
Feel free to fork and make something awesome!

## License
[GNU](https://www.gnu.org/licenses/gpl-3.0.en.html)


# kCrawler: File Management

kCrawler is an automation tool for seamlessly compressing large numbers of videos.

## Installation

The installation for KSS should cover everything needed for kCrawler.

## Usage

### Preparing the Workspace

Locate the directory that holds your input folder.  A new folder called __program_results__ will be created in this directory once the programs starts running.

### Setting Up the Environment

Ensure you have __ffmpeg__ installed (on your system, not through pip via ffmpeg-python) and check that it is __added it to your environment variables__.

### Running the Program

Run __kCrawler.py__ and paste the path to your folder when prompted.

```python
python kCrawler.py
Path to the workspace => your/path/here
```

Videos will start appearing in the __program_results__ folder.

## Contributing
Feel free to fork and make something awesome!

## License
[GNU](https://www.gnu.org/licenses/gpl-3.0.en.html)
