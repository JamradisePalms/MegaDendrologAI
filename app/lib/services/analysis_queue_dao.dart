import 'package:sqflite/sqflite.dart';
import '../models/analysis_task.dart';
import 'db_provider.dart';
import 'package:flutter/material.dart';

class AnalysisQueueDao {
  AnalysisQueueDao();
  Future<int> addTask(String imagePath, int reportId, {bool onlyOnWifi = false}) async {
    final db = await DBProvider.instance.database;
    final id = await db.insert('analysis_queue', {
      'imagePath': imagePath,
      'reportId': reportId,
      'status': 'pending',
      'createdAt': DateTime.now().toIso8601String(),
      'retries': 0,
      'onlyOnWifi': onlyOnWifi ? 1 : 0,
    });
    return id;
  }

  Future<List<AnalysisTask>> getPendingTasks() async {
    final db = await DBProvider.instance.database;
    final rows = await db.query('analysis_queue',
        where: 'status = ?', whereArgs: ['pending'], orderBy: 'createdAt ASC');
    return rows.map((r) => AnalysisTask.fromMap(r)).toList();
  }

  Future<AnalysisTask?> getById(int id) async {
    final db = await DBProvider.instance.database;
    final rows = await db.query('analysis_queue', where: 'id = ?', whereArgs: [id]);
    if (rows.isEmpty) return null;
    return AnalysisTask.fromMap(rows.first);
  }

  Future<void> updateStatus(int id, String status) async {
    final db = await DBProvider.instance.database;
    await db.update('analysis_queue', {'status': status}, where: 'id = ?', whereArgs: [id]);
  }

  Future<void> incrementRetries(int id) async {
    final db = await DBProvider.instance.database;
    await db.rawUpdate(
      'UPDATE analysis_queue SET retries = retries + 1, status = ? WHERE id = ?',
      ['pending', id],
    );
  }

  Future<void> removeTask(int id) async {
    final db = await DBProvider.instance.database;
    await db.delete('analysis_queue', where: 'id = ?', whereArgs: [id]);
  }

  Future<List<AnalysisTask>> getAllTasks() async {
    final db = await DBProvider.instance.database;
    final rows = await db.query('analysis_queue', orderBy: 'createdAt DESC');
    return rows.map((r) => AnalysisTask.fromMap(r)).toList();
  }
  Future<void> debugPrintQueue() async {
    final tasks = await getAllTasks();
    if (tasks.isEmpty) {
      debugPrint('Очередь пустая');
      return;
    }
    debugPrint('Текущая очередь анализа:');
    for (final task in tasks) {
      debugPrint(
        'id: ${task.id}, reportId: ${task.reportId}, status: ${task.status}, retries: ${task.retries}, onlyOnWifi: ${task.onlyOnWifi}, imagePath: ${task.imagePath}'
      );
    }
  }

}
