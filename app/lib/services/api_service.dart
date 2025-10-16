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
      final uri = Uri.parse('http://89.169.189.195:8080/sendphoto/gringo');
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
    try {
      String filterPart = '';

      if (features != null && features.isNotEmpty) {
        final activeFilters = features.entries
            .where((entry) => entry.value == true)
            .map((entry) => '${entry.key}=1')
            .toList();

        if (activeFilters.isNotEmpty) {
          // Объединяем через "&"
          filterPart = activeFilters.join('&');
        }
      }

      // Если фильтров нет — оставляем пробел
      if (filterPart.isEmpty) {
        filterPart = ' ';
      }

      final url = 'http://89.169.189.195:8080/filter/gringo/$filterPart/$page';
      final uri = Uri.parse(url);

      final response = await http.get(uri);

      if (response.statusCode == 200) {
        final List<dynamic> jsonList = json.decode(response.body);

        return jsonList
            .map((item) => Report.fromJson(item as Map<String, dynamic>))
            .toList();
      } else {
        throw Exception('Ошибка запроса: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Не удалось получить отчеты: $e');
    }
  }


}
