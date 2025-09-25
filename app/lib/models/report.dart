class Report {
  final int? id;
  final String? plantName;
  final double? probability;
  final String? species;
  final String? trunkRot;
  final String? trunkHoles;
  final String? trunkCracks;
  final String? trunkDamage;
  final String? crownDamage;
  final String? fruitingBodies;
  final String? diseases;
  final double? dryBranchPercentage;
  final String? additionalInfo;
  final String? overallCondition;
  final String? imageUrl;
  final String? imagePath;
  final String? analyzedAt;

  Report({
    this.id,
    this.plantName,
    this.probability,
    this.species,
    this.trunkRot,
    this.trunkHoles,
    this.trunkCracks,
    this.trunkDamage,
    this.crownDamage,
    this.fruitingBodies,
    this.diseases,
    this.dryBranchPercentage,
    this.additionalInfo,
    this.overallCondition,
    this.imageUrl,
    this.imagePath,
    this.analyzedAt,
  });

  factory Report.fromJson(Map<String, dynamic> json) {
    return Report(
      id: json['id'] is num ? (json['id'] as num).toInt() : int.tryParse(json['id']?.toString() ?? ''),
      plantName: json['plantName']?.toString(),
      probability: json['probability'] != null
          ? (json['probability'] is num
              ? (json['probability'] as num).toDouble()
              : double.tryParse(json['probability'].toString()))
          : null,
      species: json['species']?.toString(),
      trunkRot: json['trunkRot']?.toString(),
      trunkHoles: json['trunkHoles']?.toString(),
      trunkCracks: json['trunkCracks']?.toString(),
      trunkDamage: json['trunkDamage']?.toString(),
      crownDamage: json['crownDamage']?.toString(),
      fruitingBodies: json['fruitingBodies']?.toString(),
      diseases: json['diseases']?.toString(),
      dryBranchPercentage: json['dryBranchPercentage'] != null
          ? (json['dryBranchPercentage'] is num
              ? (json['dryBranchPercentage'] as num).toDouble()
              : double.tryParse(json['dryBranchPercentage'].toString()))
          : null,
      additionalInfo: json['additionalInfo']?.toString(),
      overallCondition: json['overallCondition']?.toString(),
      imageUrl: json['imageUrl']?.toString(),
      imagePath: json['imagePath']?.toString(),
      analyzedAt: json['date']?.toString(),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      if (id != null) 'id': id,
      if (plantName != null) 'plantName': plantName,
      if (probability != null) 'probability': probability,
      if (species != null) 'species': species,
      if (trunkRot != null) 'trunkRot': trunkRot,
      if (trunkHoles != null) 'trunkHoles': trunkHoles,
      if (trunkCracks != null) 'trunkCracks': trunkCracks,
      if (trunkDamage != null) 'trunkDamage': trunkDamage,
      if (crownDamage != null) 'crownDamage': crownDamage,
      if (fruitingBodies != null) 'fruitingBodies': fruitingBodies,
      if (diseases != null) 'diseases': diseases,
      if (dryBranchPercentage != null) 'dryBranchPercentage': dryBranchPercentage,
      if (additionalInfo != null) 'additionalInfo': additionalInfo,
      if (overallCondition != null) 'overallCondition': overallCondition,
      if (imageUrl != null) 'imageUrl': imageUrl,
      if (imagePath != null) 'imagePath': imagePath,
      if (analyzedAt != null) 'analyzedAt': analyzedAt,
    };
  }
}
