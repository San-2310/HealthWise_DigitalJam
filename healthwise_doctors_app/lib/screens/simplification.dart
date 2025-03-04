import 'package:flutter/material.dart';
import 'package:google_generative_ai/google_generative_ai.dart';


class JobDescriptionSimplify extends StatefulWidget {
  const JobDescriptionSimplify({super.key,required this.originalString});
  final String originalString;

  @override
  JobDescriptionSimplifyState createState() => JobDescriptionSimplifyState();
}

class JobDescriptionSimplifyState extends State<JobDescriptionSimplify> {
  // final String originalText =
  //     "Good exposure on Automation Testing tools like Selenium with Java. Should have knowledge on latest technologies. Required Candidate profile. Good communication skills required.Should be flexible with timings";
  String simplifiedText = '';
  bool isSimplifiedWindowVisible = false;

  // Initialize Gemini
  final model = GenerativeModel(
    model: 'gemini-1.5-flash',
    apiKey: 'AIzaSyAsIWg2xV5Dv-2-IR4pZBZOKwg4wbfI6So',
  );

  Future<void> simplifyText() async {
    try {
      final prompt =
          'Simplify the job description given and present it in a way of why should you consider the job in an enthusiastic way, remove all formatting: ${widget.originalString}';

      final content = [Content.text(prompt)];
      final response = await model.generateContent(content);

      setState(() {
        simplifiedText = response.text ?? 'Unable to simplify text';
        isSimplifiedWindowVisible = true;
      });
    } catch (e) {
      debugPrint('Error simplifying text: $e');
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Failed to simplify text. Please try again.'),
        ),
      );
    }
  }

  void closeSimplifiedWindow() {
    setState(() {
      isSimplifiedWindowVisible = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    // Calculate fixed height for the simplified window
    final screenHeight = MediaQuery.of(context).size.height;
    final windowHeight = screenHeight * 0.5; // 50% of screen height

    return Scaffold(
      appBar: AppBar(
        title: const Text('Job Description Translation'),
        elevation: 2,
      ),
      body: Stack(
        children: [
          // Main content
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  widget.originalString,
                  style: const TextStyle(fontSize: 18, height: 1.5),
                ),
                const SizedBox(height: 20),
                ElevatedButton.icon(
                  onPressed: simplifyText,
                  icon: const Icon(Icons.auto_awesome),
                  label: const Text('Why should you consider this job?'),
                ),
              ],
            ),
          ),
          // Simplified text window
          AnimatedPositioned(
            duration: const Duration(milliseconds: 300),
            curve: Curves.easeOut,
            bottom: isSimplifiedWindowVisible ? 0 : -windowHeight,
            left: 0,
            right: 0,
            height: windowHeight,
            child: Material(
              elevation: 8,
              child: Container(
                decoration: BoxDecoration(
                  color: Theme.of(context).cardColor,
                  borderRadius:
                      const BorderRadius.vertical(top: Radius.circular(16)),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.1),
                      blurRadius: 10,
                      offset: const Offset(0, -2),
                    ),
                  ],
                ),
                child: Column(
                  children: [
                    // Header with close button
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 16.0, vertical: 8.0),
                      decoration: BoxDecoration(
                        border: Border(
                          bottom: BorderSide(
                            color: Theme.of(context).dividerColor,
                          ),
                        ),
                      ),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Text(
                            'What does this job entail',
                            style: TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          IconButton(
                            icon: const Icon(Icons.close),
                            onPressed: closeSimplifiedWindow,
                          ),
                        ],
                      ),
                    ),
                    // Scrollable content area
                    Expanded(
                      child: SingleChildScrollView(
                        physics: const AlwaysScrollableScrollPhysics(),
                        padding: const EdgeInsets.all(16.0),
                        child: Text(
                          simplifiedText,
                          style: const TextStyle(fontSize: 16),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}