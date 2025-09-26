// lib/services/report_service.dart
import 'package:sqflite/sqflite.dart';
import '../models/report.dart';
import 'db_provider.dart';

class ReportService {
  ReportService._privateConstructor();
  static final ReportService _instance = ReportService._privateConstructor();
  factory ReportService() => _instance;

  final Map<String, String> _allowedFeatureColumns = {
    'trunkRot': 'trunkRot',
    'trunkHoles': 'trunkHoles',
    'trunkCracks': 'trunkCracks',
    'trunkDamage': 'trunkDamage',
    'crownDamage': 'crownDamage',
    'fruitingBodies': 'fruitingBodies',
    'diseases': 'diseases',
  };

  Map<String, dynamic> _reportToMap(Report r) {
    return {
      'id': r.id,
      'plantName': r.plantName,
      'probability': r.probability,
      'species': r.species,
      'trunkRot': r.trunkRot,
      'trunkHoles': r.trunkHoles,
      'trunkCracks': r.trunkCracks,
      'trunkDamage': r.trunkDamage,
      'crownDamage': r.crownDamage,
      'fruitingBodies': r.fruitingBodies,
      'diseases': r.diseases,
      'dryBranchPercentage': r.dryBranchPercentage,
      'additionalInfo': r.additionalInfo,
      'imagePath': r.imagePath,
      'imageUrl': r.imageUrl,
    };
  }

  Report _mapToReport(Map<String, dynamic> m) {
    return Report(
      id: m['id'] as int?,
      plantName: m['plantName'] as String?,
      probability: (m['probability'] as num?)?.toDouble(),
      species: m['species'] as String?,
      trunkRot: m['trunkRot'] as String?,
      trunkHoles: m['trunkHoles'] as String?,
      trunkCracks: m['trunkCracks'] as String?,
      trunkDamage: m['trunkDamage'] as String?,
      crownDamage: m['crownDamage'] as String?,
      fruitingBodies: m['fruitingBodies'] as String?,
      diseases: m['diseases'] as String?,
      dryBranchPercentage: (m['dryBranchPercentage'] as num?)?.toDouble(),
      additionalInfo: m['additionalInfo'] as String?,
      imagePath: m['imagePath'] as String?,
      imageUrl: m['imageUrl'] as String?,
    );
  }


  Future<void> init() async {
    await DBProvider.instance.database;
  }

  Future<int> saveReport(Report report) async {
    final db = await DBProvider.instance.database;
    // Возвращаем id вставленной строки
    return await db.insert('reports', _reportToMap(report),
        conflictAlgorithm: ConflictAlgorithm.replace);
  }

  /// Построение where-условий и аргументов безопасно (параметры).
  Map<String, dynamic> _buildWhere({
    double? minProbability,
    double? maxProbability,
    String? species,
    Map<String, bool>? features,
  }) {
    final List<String> parts = [];
    final List<dynamic> args = [];

    if (minProbability != null) {
      parts.add('probability >= ?');
      args.add(minProbability);
    }
    if (maxProbability != null) {
      parts.add('probability <= ?');
      args.add(maxProbability);
    }
    if (species != null) {
      parts.add('species = ?');
      args.add(species);
    }
    if (features != null) {
      features.forEach((key, value) {
        if (value == true && _allowedFeatureColumns.containsKey(key)) {
          final col = _allowedFeatureColumns[key]!;
          parts.add('$col = ?');
          args.add('Да'); // в твоей модели используешь 'Да' / 'Нет'
        }
      });
    }

    return {
      'where': parts.isEmpty ? null : parts.join(' AND '),
      'whereArgs': args,
    };
  }

  Future<List<Report>> fetchReports({
    int page = 1,
    int limit = 10,
    double? minProbability,
    double? maxProbability,
    String? species,
    Map<String, bool>? features,
  }) async {
    final db = await DBProvider.instance.database;

    final whereData = _buildWhere(
      minProbability: minProbability,
      maxProbability: maxProbability,
      species: species,
      features: features,
    );

    final String? where = whereData['where'] as String?;
    final List<dynamic> whereArgs = List<dynamic>.from(whereData['whereArgs'] as List<dynamic>? ?? []);

    final offset = (page - 1) * limit;

    final List<Map<String, dynamic>> maps = await db.query(
      'reports',
      where: where,
      whereArgs: whereArgs.isEmpty ? null : whereArgs,
      orderBy: 'id DESC', // последние добавленные первыми, как в текущей имитации
      limit: limit,
      offset: offset,
    );

    return maps.map((m) => _mapToReport(m)).toList();
  }

  Future<int> countReports({
    double? minProbability,
    double? maxProbability,
    String? species,
    Map<String, bool>? features,
  }) async {
    final db = await DBProvider.instance.database;
    final whereData = _buildWhere(
      minProbability: minProbability,
      maxProbability: maxProbability,
      species: species,
      features: features,
    );

    final String? where = whereData['where'] as String?;
    final List<dynamic> whereArgs = List<dynamic>.from(whereData['whereArgs'] as List<dynamic>? ?? []);

    final result = await db.rawQuery(
      'SELECT COUNT(*) as cnt FROM reports' + (where != null ? ' WHERE $where' : ''),
      whereArgs,
    );

    final int count = result.isNotEmpty ? (result.first['cnt'] as int) : 0;
    return count;
  }

  Future<Report?> fetchReportById(int id) async {
    final db = await DBProvider.instance.database;
    final maps = await db.query(
      'reports',
      where: 'id = ?',
      whereArgs: [id],
      limit: 1,
    );
    if (maps.isNotEmpty) {
      return _mapToReport(maps.first);
    }
    return null;
  }

  Future<void> deleteReportById(int id) async {
    final db = await DBProvider.instance.database;
    await db.delete('reports', where: 'id = ?', whereArgs: [id]);
  }
}
