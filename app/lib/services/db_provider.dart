// lib/services/db_provider.dart
import 'dart:io';
import 'package:path/path.dart';
import 'package:sqflite/sqflite.dart';
import 'package:sqflite_common_ffi/sqflite_ffi.dart';

class DBProvider {
  DBProvider._();
  static final DBProvider instance = DBProvider._();

  static Database? _database;

  Future<Database> get database async {
    if (_database != null) return _database!;
    _database = await _initDB('reports.db');
    return _database!;
  }

  Future<Database> _initDB(String fileName) async {
    if (Platform.isWindows || Platform.isLinux || Platform.isMacOS) {
      // desktop: инициализация ffi
      sqfliteFfiInit();
      final dbFactory = databaseFactoryFfi;
      final path = join(await databaseFactoryFfi.getDatabasesPath(), fileName);
      return await dbFactory.openDatabase(
        path,
        options: OpenDatabaseOptions(
          version: 1,
          onCreate: _onCreate,
        ),
      );
    } else {
      // mobile (Android/iOS): обычный sqflite
      final dbPath = await getDatabasesPath();
      final path = join(dbPath, fileName);
      return await openDatabase(
        path,
        version: 1,
        onCreate: _onCreate,
      );
    }
  }

  Future _onCreate(Database db, int version) async {
    await db.execute('''
      CREATE TABLE reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plantName TEXT,
        probability REAL,
        species TEXT,
        trunkRot TEXT,
        trunkHoles TEXT,
        trunkCracks TEXT,
        trunkDamage TEXT,
        crownDamage TEXT,
        fruitingBodies TEXT,
        diseases TEXT,
        dryBranchPercentage REAL,
        additionalInfo TEXT,
        imagePath TEXT
      )
    ''');

    // индексы для ускорения фильтров
    await db.execute('CREATE INDEX idx_reports_species ON reports(species)');
    await db.execute('CREATE INDEX idx_reports_probability ON reports(probability)');
    await db.execute('CREATE INDEX idx_reports_trunkRot ON reports(trunkRot)');
  }

  Future close() async {
    final db = _database;
    if (db != null) {
      await db.close();
      _database = null;
    }
  }
}
