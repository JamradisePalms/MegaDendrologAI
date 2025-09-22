class Report {
  final String plantName;
  final double probability;
  final String species;
  final String trunkRot;
  final String trunkHoles;
  final String trunkCracks;
  final String trunkDamage;
  final String crownDamage;
  final String fruitingBodies;
  final String diseases;
  final double dryBranchPercentage;
  final String additionalInfo;

  Report({
    required this.plantName,
    required this.probability,
    required this.species,
    required this.trunkRot,
    required this.trunkHoles,
    required this.trunkCracks,
    required this.trunkDamage,
    required this.crownDamage,
    required this.fruitingBodies,
    required this.diseases,
    required this.dryBranchPercentage,
    required this.additionalInfo,
  });
}
