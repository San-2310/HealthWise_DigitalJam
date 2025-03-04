// import 'dart:convert';

// import 'package:flutter/material.dart';
// import 'package:wear_plus/wear_plus.dart';
// import 'package:web_socket_channel/io.dart';
// import 'package:web_socket_channel/web_socket_channel.dart';
// class WatchHomeScreen extends StatefulWidget {
//   @override
//   _WatchHomeScreenState createState() => _WatchHomeScreenState();
// }

// class _WatchHomeScreenState extends State<WatchHomeScreen> {
//   late WebSocketChannel channel;
//   int heartRate = 0;
//   int bloodPressureSystolic = 0;
//   int bloodPressureDiastolic = 0;
//   int spo2 = 0;
//   int temperature = 0;

//   @override
//   void initState() {
//     super.initState();
//     channel = IOWebSocketChannel.connect('ws://10.21.9.214:8080');
//     channel.stream.listen((message) {
//       final data = jsonDecode(message);
//       setState(() {
//         heartRate = data['heart_rate'];
//         bloodPressureSystolic = data['blood_pressure_systolic'];
//         bloodPressureDiastolic = data['blood_pressure_diastolic'];
//         spo2 = data['spo2'];
//         temperature = data['temperature'];
//       });
//     });
//   }

//   @override
//   void dispose() {
//     channel.sink.close();
//     super.dispose();
//   }

//   @override
//   Widget build(BuildContext context) {
//     return WatchShape(
//       builder: (BuildContext context, WearShape shape, Widget? child) {
//         return AmbientMode(
//           builder: (context, mode, child) {
//             return mode == WearMode.active ? ActiveWatchFace(heartRate, bloodPressureSystolic, bloodPressureDiastolic, spo2, temperature) : AmbientWatchFace();
//           },
//         );
//       },
//     );
//   }
// }

// class ActiveWatchFace extends StatelessWidget {
//   final int heartRate;
//   final int bloodPressureSystolic;
//   final int bloodPressureDiastolic;
//   final int spo2;
//   final int temperature;

//   ActiveWatchFace(this.heartRate, this.bloodPressureSystolic, this.bloodPressureDiastolic, this.spo2, this.temperature);

//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       body: Center(
//         child: GridView(
//           gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(crossAxisCount: 2),
//           padding: EdgeInsets.all(8.0),
//           children: [
//             _buildInfoTile("Heart Rate", "$heartRate bpm"),
//             _buildInfoTile("Blood Pressure", "$bloodPressureSystolic/$bloodPressureDiastolic"),
//             _buildInfoTile("Oxygen Saturation", "$spo2%"),
//             _buildInfoTile("Temperature", "$temperature°C"),
//           ],
//         ),
//       ),
//     );
//   }

//   Widget _buildInfoTile(String label, String value) {
//     return Card(
//       shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
//       color: Colors.blueGrey[900],
//       elevation: 4,
//       child: Padding(
//         padding: const EdgeInsets.all(12.0),
//         child: Column(
//           mainAxisAlignment: MainAxisAlignment.center,
//           children: [
//             Text(
//               label,
//               style: TextStyle(color: Colors.white70, fontSize: 12),
//               textAlign: TextAlign.center,
//             ),
//             SizedBox(height: 8),
//             Text(
//               value,
//               style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold),
//               textAlign: TextAlign.center,
//             ),
//           ],
//         ),
//       ),
//     );
//   }
// }

// class AmbientWatchFace extends StatelessWidget {
//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       body: Center(
//         child: Text(
//           "Ambient Mode",
//           style: TextStyle(color: Colors.white70, fontSize: 14),
//         ),
//       ),
//     );
//   }
// }


import 'dart:math';
import 'package:flutter/material.dart';
import 'package:healthwise_watch_os/calm_now.dart';
import 'package:healthwise_watch_os/notifications.dart';
import 'package:wear_plus/wear_plus.dart';

class WatchHomeScreen extends StatefulWidget {
  @override
  _WatchHomeScreenState createState() => _WatchHomeScreenState();
}

class _WatchHomeScreenState extends State<WatchHomeScreen> {
  int heartRate = 75;
  int bloodPressureSystolic = 120;
  int bloodPressureDiastolic = 80;
  int spo2 = 98;
  int temperature = 37;

  @override
  void initState() {
    super.initState();
    _generateDummyData();
  }

  void _generateDummyData() {
    // Start generating dummy data periodically
    Future.delayed(Duration(seconds: 1), () {
      setState(() {
        heartRate = 60 + Random().nextInt(40);
        bloodPressureSystolic = 110 + Random().nextInt(20);
        bloodPressureDiastolic = 70 + Random().nextInt(15);
        spo2 = 95 + Random().nextInt(4);
        temperature = 36 + Random().nextInt(3);
      });
      _generateDummyData(); // Re-run the function to keep updating
    });
  }

  @override
  Widget build(BuildContext context) {
    return WatchShape(
      builder: (BuildContext context, WearShape shape, Widget? child) {
        return AmbientMode(
          builder: (context, mode, child) {
            return mode == WearMode.active
                ? ActiveWatchFace(
                    heartRate, bloodPressureSystolic, bloodPressureDiastolic, spo2, temperature)
                : AmbientWatchFace();
          },
        );
      },
    );
  }
}

class ActiveWatchFace extends StatelessWidget {
  final int heartRate;
  final int bloodPressureSystolic;
  final int bloodPressureDiastolic;
  final int spo2;
  final int temperature;
  
 

  ActiveWatchFace(
      this.heartRate, this.bloodPressureSystolic, this.bloodPressureDiastolic, this.spo2, this.temperature);

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;
     print(heartRate);
    return Scaffold(
      appBar: AppBar(
        title: Text("Healthwise", style: TextStyle(fontSize: 12)),
        actions: [
          IconButton(
            onPressed: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => AllNotificationsScreen()),
              );
            },
            icon: Icon(Icons.notifications, size: 12),
          ),
        ],
      ),
      body: SingleChildScrollView(
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              _buildInfoTile("Heart Rate", "$heartRate bpm", size),
              _buildInfoTile("Blood Pressure", "$bloodPressureSystolic/$bloodPressureDiastolic", size),
              _buildInfoTile("Oxygen Saturation", "$spo2%", size),
              _buildInfoTile("Temperature", "$temperature°C", size),
              ElevatedButton(
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (context) => CalmNowScreen()),
                  );
                },
                child: Text("Calm Now"),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildInfoTile(String label, String value, Size size) {
    return Container(
      width: size.width * 0.8,
      padding: EdgeInsets.symmetric(vertical: 5),
      child: Card(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
        color: Colors.blueGrey[900],
        elevation: 4,
        child: Padding(
          padding: const EdgeInsets.all(12.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(
                label,
                style: TextStyle(color: Colors.white70, fontSize: 12),
                textAlign: TextAlign.center,
              ),
              SizedBox(height: 8),
              Text(
                value,
                style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold),
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class AmbientWatchFace extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Text(
          "Ambient Mode",
          style: TextStyle(color: Colors.white70, fontSize: 14),
        ),
      ),
    );
  }
}
