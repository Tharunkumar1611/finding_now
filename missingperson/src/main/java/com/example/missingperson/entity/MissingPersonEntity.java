package com.example.missingperson.entity;

import jakarta.persistence.*;
import lombok.*;

@Entity
@Table(name = "MissingPersons")
@NoArgsConstructor
@Getter
@Setter
public class MissingPersonEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String name;

    @Column(nullable = false)
    private int age;

    @Column(nullable = false)
    private String gender;

    @Column(nullable = false)
    private String lastSeenLocation;

    @Lob
    private byte[] image;

    @Column
    private int reportedBy;  // Reference to official_id
}
