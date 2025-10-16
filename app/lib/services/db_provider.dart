// lib/services/db_provider.dart
import 'dart:io';
import 'package:path/path.dart';
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
          version: 3, // ⬅️ поднял версию (чтобы миграция сработала)
          onCreate: _onCreate,
          onUpgrade: _onUpgrade,
        ),
      );
    } else {
      // mobile (Android/iOS): обычный sqflite
      final dbPath = await getDatabasesPath();
      final path = join(dbPath, fileName);
      return await openDatabase(
        path,
        version: 3, // ⬅️ тоже версия 3
        onCreate: _onCreate,
        onUpgrade: _onUpgrade,
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
      overallCondition TEXT,
      imageUrl TEXT,
      imagePath TEXT,
      analyzedAt TEXT,
      isVerified INTEGER DEFAULT 0
    )
  ''');

    // индексы для ускорения фильтров
    await db.execute('CREATE INDEX idx_reports_species ON reports(species)');
    await db.execute('CREATE INDEX idx_reports_probability ON reports(probability)');
    await db.execute('CREATE INDEX idx_reports_trunkRot ON reports(trunkRot)');

     await db.execute('''
        CREATE TABLE analysis_queue (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        imagePath TEXT NOT NULL,
        reportId INTEGER NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending', -- pending|uploading|done|failed
        createdAt TEXT NOT NULL,
        retries INTEGER NOT NULL DEFAULT 0,
        onlyOnWifi INTEGER NOT NULL DEFAULT 0
      )
      ''');
      await db.execute('CREATE INDEX idx_queue_status ON analysis_queue(status)');
    }



  Future _onUpgrade(Database db, int oldVersion, int newVersion) async {
    if (oldVersion < 2) {
      // добавляем новые колонки
      await db.execute('ALTER TABLE reports ADD COLUMN overallCondition TEXT');
      await db.execute('ALTER TABLE reports ADD COLUMN imageUrl TEXT');
      await db.execute('ALTER TABLE reports ADD COLUMN analyzedAt TEXT');
    }
    if (oldVersion < 3) {
    // добавляем isVerified в reports
    await db.execute('ALTER TABLE reports ADD COLUMN isVerified INTEGER DEFAULT 0');
    // создаём таблицу очереди если ранее не было
    await db.execute('''
      CREATE TABLE IF NOT EXISTS analysis_queue (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      imagePath TEXT NOT NULL,
      reportId INTEGER NOT NULL,
      status TEXT NOT NULL DEFAULT 'pending',
      createdAt TEXT NOT NULL,
      retries INTEGER NOT NULL DEFAULT 0,
      onlyOnWifi INTEGER NOT NULL DEFAULT 0
    );
    ''');
    await db.execute('CREATE INDEX IF NOT EXISTS idx_queue_status ON analysis_queue(status)');
    }
  }
  

  Future close() async {
    final db = _database;
    if (db != null) {
      await db.close();
      _database = null;
    }
  }

}