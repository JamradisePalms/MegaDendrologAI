import 'dart:io';
import 'package:flutter/material.dart';

class DisplayScreen extends StatelessWidget {
  final File imageFile;
  const DisplayScreen({super.key, required this.imageFile});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Просмотр фото')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Image.file(imageFile),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Назад'),
            ),
          ],
        ),
      ),
    );
  }
}
