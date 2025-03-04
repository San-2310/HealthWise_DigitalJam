import 'package:flutter/material.dart';
import 'package:netwealth_vjti/models/populate_users.dart';
import 'package:netwealth_vjti/models/doctor.dart' as ModelUser;
import 'package:netwealth_vjti/resources/user_provider.dart';
import 'package:netwealth_vjti/screens/api_marketplace.dart';
import 'package:netwealth_vjti/screens/chat_screen.dart/chat_screen.dart';
import 'package:netwealth_vjti/screens/chat_screen.dart/search_screen.dart';
import 'package:netwealth_vjti/screens/image_to_text.dart';
import 'package:netwealth_vjti/screens/job_matching.dart';
import 'package:netwealth_vjti/screens/networking_screen.dart';
import 'package:netwealth_vjti/screens/news_app_screen.dart';
import 'package:netwealth_vjti/screens/posts_screen/feed_screen.dart';
import 'package:netwealth_vjti/screens/projects/projects_screen.dart';
import 'package:netwealth_vjti/screens/user_profile.dart';
import 'package:netwealth_vjti/widgets/app_drawer.dart';
import 'package:provider/provider.dart';


class MainLayoutScreen extends StatefulWidget {
  @override
  _MainLayoutScreenState createState() => _MainLayoutScreenState();
}

class _MainLayoutScreenState extends State<MainLayoutScreen> {
  int _selectedIndex = 0;

  // List of pages for each BottomNavigationBar item
  final List<Widget> _pages = [
    FeedScreen(),
    NetworkingScreen(),
    JobMatchingScreen(),
    ViewProjectsScreen(),
    UserProfileScreen(userId: "lUoJzE4iXrVAvJudsCHwUwVjfPB2",),
  ];

  // Function to handle item selection
  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _pages[_selectedIndex], // Display the current page
      bottomNavigationBar: BottomNavigationBar(
        selectedItemColor: const Color.fromRGBO(55, 27, 52, 1),
        unselectedItemColor: const Color.fromRGBO(205, 208, 227, 1),
        currentIndex: _selectedIndex, // Current selected index
        onTap: _onItemTapped, // Update index on item tap
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.home),
            label: 'Home',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.group),
            label: 'Network',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.medical_services_outlined),
            label: 'Jobs',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.book),
            label: 'Projects',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.person),
            label: 'Profile',
          ),
        ],
      ),
    );
  }
}
