import 'dart:async';
import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/material.dart';

class MusicAppScreen extends StatefulWidget {
  const MusicAppScreen({Key? key}) : super(key: key);

  @override
  _MusicAppScreenState createState() => _MusicAppScreenState();
}

class _MusicAppScreenState extends State<MusicAppScreen> {
  AudioPlayer _audioPlayer = AudioPlayer();
  var tabs = [];
  int currentTabIndex = 0;
  bool isPlaying = false;
  Music? music;

  Widget miniPlayer(Music? music, {bool stop = false}) {
    this.music = music;
    if (music == null) return SizedBox();
    if (stop) {
      isPlaying = false;
      _audioPlayer.stop();
    }
    setState(() {});
    Size deviceSize = MediaQuery.of(context).size;
    return AnimatedContainer(
      duration: const Duration(milliseconds: 500),
      color: Colors.lightBlueAccent,
      width: deviceSize.width,
      height: 50,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Image.network(music.image, fit: BoxFit.cover),
          Expanded(
            child: Text(
              music.name,
              style: TextStyle(color: Colors.white, fontSize: 18),
              overflow: TextOverflow.ellipsis,
            ),
          ),
          IconButton(
            onPressed: () async {
              isPlaying = !isPlaying;
              if (isPlaying) {
                await _audioPlayer.play(AssetSource(music.audioURL));
              } else {
                await _audioPlayer.pause();
              }
              setState(() {});
            },
            icon: Icon(
              isPlaying ? Icons.pause : Icons.play_arrow,
              color: Colors.white,
            ),
          ),
        ],
      ),
    );
  }

  @override
  initState() {
    super.initState();
    tabs = [Home(miniPlayer)];
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        title: const Text('Music'),
        actions: [SizedBox(width: 10)],
      ),
      body: tabs[currentTabIndex],
      backgroundColor: Colors.grey[200],
      bottomNavigationBar: Column(
        mainAxisSize: MainAxisSize.min,
        children: [miniPlayer(music)],
      ),
    );
  }
}

class Home extends StatelessWidget {
  final Function _miniPlayer;
  Home(this._miniPlayer);

  Widget createCategory(Category category) {
    return Container(
      color: Colors.blueGrey[50],
      child: Row(
        children: [
          SizedBox(width: 4),
          Image.network(category.imageURL, width: 50, height: 50),
          Padding(
            padding: EdgeInsets.only(left: 8),
            child: Text(
              category.name,
              style: TextStyle(color: Colors.black),
            ),
          ),
        ],
      ),
    );
  }

  List<Widget> createListOfCategories() {
    List<Category> categoryList = CategoryOperations.getCategories();
    return categoryList
        .map((Category category) => createCategory(category))
        .toList();
  }

  Widget createMusic(Music music) {
    return Padding(
      padding: EdgeInsets.all(8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            height: 150,
            width: 150,
            child: InkWell(
              onTap: () {
                _miniPlayer(music, stop: true);
              },
              child: Image.network(music.image, fit: BoxFit.cover),
            ),
          ),
          Text(music.name, style: TextStyle(color: Colors.black)),
          Text(music.desc, style: TextStyle(color: Colors.black54)),
        ],
      ),
    );
  }

  Widget createMusicList(String label) {
    List<Music> musicList = MusicOperations.getMusic();
    return Padding(
      padding: EdgeInsets.only(left: 10),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: TextStyle(color: Colors.black, fontSize: 18, fontWeight: FontWeight.bold),
          ),
          Container(
            height: 250,
            child: ListView.builder(
              scrollDirection: Axis.horizontal,
              itemBuilder: (ctx, index) {
                return createMusic(musicList[index]);
              },
              itemCount: musicList.length,
            ),
          ),
        ],
      ),
    );
  }

  Widget createGrid() {
    return Container(
      padding: EdgeInsets.all(8),
      height: 250,
      child: GridView.count(
        crossAxisCount: 2,
        childAspectRatio: 5 / 2,
        crossAxisSpacing: 8,
        mainAxisSpacing: 8,
        children: createListOfCategories(),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      child: SafeArea(
        child: Container(
          child: Column(
            children: [
              createGrid(),
              createMusicList('Made for You'),
              createMusicList('Popular Playlist'),
            ],
          ),
        ),
      ),
    );
  }
}

class Music {
  String name;
  String image;
  String desc;
  String audioURL;
  Music(this.name, this.image, this.desc, this.audioURL);
}

class Category {
  String name;
  String imageURL;
  Category(this.name, this.imageURL);
}

class CategoryOperations {
  CategoryOperations._() {}

  static List<Category> getCategories() {
    return <Category>[
      Category('Piano Sounds', 'https://linktoimage.jpg'),
      Category('Guitar', 'https://linktoimage.jpg'),
      Category('Violin', 'https://linktoimage.jpg'),
      Category('Flute', 'https://linktoimage.jpg'),
      Category('Nature', 'https://linktoimage.jpg'),
      Category('Rain', 'https://linktoimage.jpg'),
    ];
  }
}

class MusicOperations {
  MusicOperations._() {}

  static List<Music> getMusic() {
    return <Music>[
      Music('Emotional Piano Music', 'https://linktoimage.jpg', 'SigmaMusicArt', 'songs/song1.mp3'),
      Music('Inspirational Uplifting Calm Piano', 'https://linktoimage.jpg', 'Leberchmus', 'songs/song2.mp3'),
      Music('Relaxing Piano Music', 'https://linktoimage.jpg', 'Clavier-Music', 'songs/song3.mp3'),
      Music('Tibet', 'https://linktoimage.jpg', 'Top-Flow', 'songs/song4.mp3'),
    ];
  }
}

class AudioPlayerScreen extends StatefulWidget {
  const AudioPlayerScreen();

  @override
  _AudioPlayerScreenState createState() => _AudioPlayerScreenState();
}

class _AudioPlayerScreenState extends State<AudioPlayerScreen> {
  late AudioPlayer player;

  @override
  void initState() {
    super.initState();
    player = AudioPlayer();
    player.setReleaseMode(ReleaseMode.stop);

    WidgetsBinding.instance.addPostFrameCallback((_) async {
      try {
        await player.setSource(AssetSource('songs/song1.mp3'));
        await player.resume();
      } catch (e) {
        print('Error initializing audio player: $e');
      }
    });
  }

  @override
  void dispose() {
    player.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Audio Player')),
      body: PlayerWidget(player: player),
    );
  }
}

class PlayerWidget extends StatefulWidget {
  final AudioPlayer player;
  const PlayerWidget({required this.player, super.key});

  @override
  _PlayerWidgetState createState() => _PlayerWidgetState();
}

class _PlayerWidgetState extends State<PlayerWidget> {
  PlayerState? _playerState;
  Duration? _duration;
  Duration? _position;

  bool get _isPlaying => _playerState == PlayerState.playing;
  bool get _isPaused => _playerState == PlayerState.paused;

  String get _durationText => _duration?.toString().split('.').first ?? '';
  String get _positionText => _position?.toString().split('.').first ?? '';

  AudioPlayer get player => widget.player;

  @override
  void initState() {
    super.initState();
    _playerState = player.state;
    player.getDuration().then((value) => setState(() {
          _duration = value;
        }));
    player.getCurrentPosition().then((value) => setState(() {
          _position = value;
        }));
    _initStreams();
  }

  void _initStreams() {
    player.onDurationChanged.listen((duration) => setState(() => _duration = duration));
    player.onPositionChanged.listen((p) => setState(() => _position = p));
    player.onPlayerComplete.listen((event) {
      setState(() {
        _playerState = PlayerState.stopped;
        _position = Duration.zero;
      });
    });
    player.onPlayerStateChanged.listen((state) {
      setState(() {
        _playerState = state;
      });
    });
  }

  Future<void> _play() async {
    await player.resume();
    setState(() => _playerState = PlayerState.playing);
  }

  Future<void> _pause() async {
    await player.pause();
    setState(() => _playerState = PlayerState.paused);
  }

  Future<void> _stop() async {
    await player.stop();
    setState(() {
      _playerState = PlayerState.stopped;
      _position = Duration.zero;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: <Widget>[
        Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            IconButton(
              onPressed: _isPlaying ? null : _play,
              iconSize: 48.0,
              icon: const Icon(Icons.play_arrow),
            ),
            IconButton(
              onPressed: _isPlaying ? _pause : null,
              iconSize: 48.0,
              icon: const Icon(Icons.pause),
            ),
            IconButton(
              onPressed: _isPlaying || _isPaused ? _stop : null,
              iconSize: 48.0,
              icon: const Icon(Icons.stop),
            ),
          ],
        ),
        Slider(
          onChanged: (value) {
            final duration = _duration;
            if (duration == null) return;
            final position = value * duration.inMilliseconds;
            player.seek(Duration(milliseconds: position.round()));
          },
          value: (_position != null && _duration != null &&
              _position!.inMilliseconds > 0 &&
              _position!.inMilliseconds < _duration!.inMilliseconds)
              ? _position!.inMilliseconds / _duration!.inMilliseconds
              : 0.0,
        ),
        Text(
          _position != null ? '$_positionText / $_durationText' : _duration != null ? _durationText : '',
          style: TextStyle(fontSize: 16.0),
        ),
      ],
    );
  }
}