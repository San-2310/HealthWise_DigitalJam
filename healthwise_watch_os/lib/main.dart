import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:healthwise_watch_os/firebase_options.dart';
import 'package:healthwise_watch_os/home.dart';

void main() async {
  // WidgetsFlutterBinding.ensureInitialized();
  // await Firebase.initializeApp(
  //   options: DefaultFirebaseOptions.currentPlatform,
  // );
  //NotificationService.initialize();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: WatchHomeScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class NotificationService {
  static late FlutterLocalNotificationsPlugin _flutterLocalNotificationsPlugin;
  static final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  static List<NotificationModel> _notifications = [];

  // Initialize the notification plugin
  static void initialize() async {
    _flutterLocalNotificationsPlugin = FlutterLocalNotificationsPlugin();

    const AndroidInitializationSettings initializationSettingsAndroid =
        AndroidInitializationSettings('@mipmap/ic_launcher');
    final InitializationSettings initializationSettings =
        InitializationSettings(android: initializationSettingsAndroid);
    await _flutterLocalNotificationsPlugin.initialize(initializationSettings);

    _listenForNewNotifications();
  }

  // Listen for new notifications from Firestore
  static void _listenForNewNotifications() {
    _firestore.collection('notifications').orderBy('timestamp', descending: true).snapshots().listen((snapshot) {
      if (snapshot.docs.isNotEmpty) {
        NotificationModel notification = NotificationModel.fromFirestore(snapshot.docs.last.data());
        // Add the new notification to the list
        _notifications.insert(0, notification);  // Insert at the beginning of the list
        // Send local notification
        sendLocalNotification(notification);
      }
    });
  }

  // Send a local notification when a new notification is added
  static void sendLocalNotification(NotificationModel notification) async {
    const AndroidNotificationDetails androidPlatformChannelSpecifics =
        AndroidNotificationDetails(
      'default_channel',
      'Default',
      channelDescription: 'Channel for basic notifications',
      importance: Importance.max,
      priority: Priority.high,
      showWhen: false,
    );

    const NotificationDetails platformChannelSpecifics = NotificationDetails(
      android: androidPlatformChannelSpecifics,
    );

    await _flutterLocalNotificationsPlugin.show(
      0,
      notification.title,
      notification.message,
      platformChannelSpecifics,
      payload: 'New notification clicked',
    );
  }
}

// Notification Model to handle Firestore data
class NotificationModel {
  final String title;
  final String message;
  final DateTime timestamp;

  NotificationModel({required this.title, required this.message, required this.timestamp});

  // Convert Firestore data to NotificationModel
  factory NotificationModel.fromFirestore(Map<String, dynamic> firestoreDoc) {
    return NotificationModel(
      title: firestoreDoc['title'],
      message: firestoreDoc['message'],
      timestamp: (firestoreDoc['timestamp'] as Timestamp).toDate(),
    );
  }
}