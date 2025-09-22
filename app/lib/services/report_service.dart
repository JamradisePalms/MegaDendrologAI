import 'dart:async';
import 'package:plant_analyzer/models/report.dart';

class ReportService {
  ReportService._privateConstructor();
  static final ReportService _instance = ReportService._privateConstructor();
  factory ReportService() => _instance;
  // Эмуляция офлайн-кэша
  final List<Report> _cachedReports = List.generate(
    25,
    (index) => Report(
      plantName: 'Plant $index',
      probability: 50 + index.toDouble(),
      species: 'oak',
      trunkRot: 'Нет',
      trunkHoles: 'Да',
      trunkCracks: 'Нет',
      trunkDamage: 'Нет',
      crownDamage: 'Нет',
      fruitingBodies: 'Нет',
      diseases: 'Нет',
      dryBranchPercentage: index.toDouble(),
      additionalInfo: 'Some info $index',
    ),
  );
  Future<void> saveReport(Report report) async {
    _cachedReports.insert(0, report);
  }
  Future<int> countReports({
      double? minProbability,
      double? maxProbability,
      String? species,
      Map<String, bool>? features,
    }) async {
      await Future.delayed(Duration(milliseconds: 100));
      return _cachedReports.where((r) {
        if (minProbability != null && r.probability < minProbability) return false;
        if (maxProbability != null && r.probability > maxProbability) return false;
        if (species != null && r.species != species) return false;
        if (features != null) {
          for (var entry in features.entries) {
            if (entry.value) {
              switch (entry.key) {
                case 'trunkRot':
                  if (r.trunkRot != 'Да') return false;
                  break;
                case 'trunkHoles':
                  if (r.trunkHoles != 'Да') return false;
                  break;
                case 'trunkCracks':
                  if (r.trunkCracks != 'Да') return false;
                  break;
                case 'trunkDamage':
                  if (r.trunkDamage != 'Да') return false;
                  break;
                case 'crownDamage':
                  if (r.crownDamage != 'Да') return false;
                  break;
                case 'fruitingBodies':
                  if (r.fruitingBodies != 'Да') return false;
                  break;
                case 'diseases':
                  if (r.diseases != 'Да') return false;
                  break;
              }
            }
          }
        }
        return true;
      }).length;
    }
  Future<List<Report>> fetchReports({
    int page = 1,
    int limit = 10,
    double? minProbability,
    double? maxProbability,
    String? species,
    Map<String, bool>? features,
  }) async {
    // имитация задержки сети
    await Future.delayed(Duration(milliseconds: 500));

    // фильтрация
    List<Report> filtered = _cachedReports.where((r) {
      if (minProbability != null && r.probability < minProbability) return false;
      if (maxProbability != null && r.probability > maxProbability) return false;
      if (species != null && r.species != species) return false;
      if (features != null) {
        if (features['trunkRot'] == true && r.trunkRot != 'Да') return false;
        if (features['trunkHoles'] == true && r.trunkHoles != 'Да') return false;
        if (features['trunkCracks'] == true && r.trunkCracks != 'Да') return false;
        if (features['trunkDamage'] == true && r.trunkDamage != 'Да') return false;
        if (features['crownDamage'] == true && r.crownDamage != 'Да') return false;
        if (features['fruitingBodies'] == true && r.fruitingBodies != 'Да') return false;
        if (features['diseases'] == true && r.diseases != 'Да') return false;
      }
      return true;
    }).toList();

    // пагинация
    int start = (page - 1) * limit;
    int end = start + limit;
    if (start >= filtered.length) return [];
    return filtered.sublist(start, end > filtered.length ? filtered.length : end);
  }

  Future<Report?> fetchReportById(int id) async {
    await Future.delayed(Duration(milliseconds: 300));
    if (id >= 0 && id < _cachedReports.length) return _cachedReports[id];
    return null;
  }
}
