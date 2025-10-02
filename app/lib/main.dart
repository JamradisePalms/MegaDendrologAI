import 'dart:io';
import 'package:flutter/material.dart';
import 'package:plant_analyzer/theme.dart';
import 'package:sqflite_common_ffi/sqflite_ffi.dart';
import 'screens/home_screen.dart';
import 'services/queue_sync_service.dart';
import 'services/db_provider.dart';
import 'theme.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Инициализация SQLite для desktop
  if (Platform.isWindows || Platform.isLinux || Platform.isMacOS) {
    sqfliteFfiInit();
    databaseFactory = databaseFactoryFfi;
  }
  await DBProvider.instance.database;
  await QueueSyncService.instance.start();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'PlantGuard',
      theme: appTheme,
      home: const HomeScreen(),
    );
  }
}
