import 'dart:io';
import 'dart:async';
import 'dart:convert';
import 'package:flutter/services.dart' show rootBundle;
import '../models/report.dart';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';

class ApiService {
  Future<List<Report>> analyzeImage(File imageFile) async {
    List<Report> reports = [];
    try {
      final uri = Uri.parse('http://51.250.109.178:8080/sendphoto/1');
      final request = http.MultipartRequest('POST', uri);

      request.files.add(await http.MultipartFile.fromPath(
        'file',
        imageFile.path,
        contentType: MediaType('image', 'png'),
      ));

      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        debugPrint('Ответ API: ${response.body}');

        final decoded = json.decode(response.body);

        if (decoded is List) {
          // если сервер возвращает массив
          reports = decoded
              .map((item) => Report.fromJson(item as Map<String, dynamic>))
              .toList();
        } else if (decoded is Map<String, dynamic>) {
          // если сервер может вернуть один объект
          reports = [Report.fromJson(decoded)];
        }

        for (final r in reports) {
          debugPrint(r.debugString());
        }
      } else {
        debugPrint('Ошибка запроса: ${response.statusCode} → ${response.body}');
      }
    } catch (e) {
      debugPrint('Ошибка запроса: $e');
    }

    return reports;
  }
  Future<List<Report>> fetchReports({
    int page = 1,
    int limit = 10,
    double? minProbability,
    double? maxProbability,
    String? species,
    Map<String, bool>? features,
  }) async {
    // Заглушка — читаем локальный JSON из assets
    final jsonString = await rootBundle.loadString('assets/mock/reports.json');
    final List<dynamic> jsonList = json.decode(jsonString);

    // Конвертируем в список Report
    final reports = jsonList
        .map((item) => Report.fromJson(item as Map<String, dynamic>))
        .toList();
    return reports;
  }
}
