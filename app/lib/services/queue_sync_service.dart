import 'dart:async';
import 'dart:io';

import 'analysis_queue_dao.dart';
import '../models/analysis_task.dart';
import '../services/api_service.dart';
import '../services/report_service.dart';
import '../services/connectivity_service.dart';

class QueueSyncService {
  static final QueueSyncService instance = QueueSyncService._();
  QueueSyncService._();

  final AnalysisQueueDao _dao = AnalysisQueueDao();
  bool _isSyncing = false;
  Timer? _timer;

  Future<void> start() async {
    // запуск синка сразу при старте
    await syncPendingTasks();

    // запуск каждые 30 секунд
    _timer ??= Timer.periodic(const Duration(seconds: 30), (_) async {
      await syncPendingTasks();
    });
  }

  Future<void> stop() async {
    _timer?.cancel();
    _timer = null;
  }

  Future<void> syncPendingTasks() async {
    if (_isSyncing) return;
    _isSyncing = true;

    try {
      final hasInternet = await ConnectivityService.hasInternet();
      if (!hasInternet) return;

      final tasks = await _dao.getPendingTasks();
      final api = ApiService();
      final reportService = ReportService();

      for (final task in tasks) {
        final file = File(task.imagePath);
        if (!file.existsSync()) {
          await _dao.updateStatus(task.id!, 'failed');
          continue;
        }

        await _dao.updateStatus(task.id!, 'uploading');

        try {
          final serverReports = await api.analyzeImage(file); // теперь список

          // заменяем локальный отчёт данными с сервера
          for (final serverReport in serverReports) {
            await reportService.replaceReportKeepingId(task.reportId, serverReport);
          }

          await _dao.updateStatus(task.id!, 'done');
          await _dao.removeTask(task.id!);
        } catch (e) {
          await _dao.incrementRetries(task.id!);
          final refreshed = await _dao.getById(task.id!);
          final retries = refreshed?.retries ?? (task.retries + 1);
          if (retries > 5) {
            await _dao.updateStatus(task.id!, 'failed');
          } else {
            await _dao.updateStatus(task.id!, 'pending');
          }
        }

      }
    } finally {
      _isSyncing = false;
    }
  }
}
